"""Acquisition strategies under a duration budget.

The question this file answers: given a pool of *unlabeled* sign language video and a
fixed number of annotator-hours, which clips should you pay to annotate?

Scope note. Annotation type is held fixed within a run: every selected clip is bought
at the same fidelity. That isolates the claim under test, which is about the
*selection signal*, not about mixing annotation types. Allocating between gloss and
translation under one budget is a separate question with a separate answer, and
conflating the two would make it impossible to attribute any gain to either.

Strategies split into two families, and the distinction matters:

  LABEL-FREE, computable on a pool nobody has touched:
    random              uniform over clips
    duration_random     uniform over annotator-seconds, not clips. The control that
                        matters: any coverage method implicitly prefers longer clips
                        (more frames, more articulatory variety), so beating plain
                        random could just mean "bought longer videos."
    phon_coverage       ours: submodular coverage over pose-derived articulatory
                        channels

  MODEL-BASED, need a seed model already trained on some labeled data:
    uncertainty         decoder entropy on the pool

Reporting both families honestly is the point. A label-free method that matches a
model-based one is more useful in practice, because it works at the moment when you
have no model and no labels, which is exactly when corpus-planning decisions get made.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from descriptors import (ChannelCoverage, ArticulatoryConfig, ArticulatoryFeaturizer)

# Annotator-seconds per second of video. Gloss is anchored on How2Sign's reported
# ~1 hour per 90 seconds. Translation is cheaper and needs a fluent bilingual rather
# than a trained glosser. Both are treated as hyperparameters.
COST_REALTIME = {"translation": 6.0, "gloss": 40.0}


def clip_cost(duration_s: float, annotation: str) -> float:
    return duration_s * COST_REALTIME[annotation]


# ---------------------------------------------------------------------------
# articulatory features for the pool
# ---------------------------------------------------------------------------

def fit_featurizer(pose_paths: Sequence[Path], n_fit: int = 1200, seed: int = 0,
                   n_hs: int = 40) -> ArticulatoryFeaturizer:
    """Fit codebooks on a sample of the pool.

    Fitting on the *unlabeled pool* is deliberate and is what makes the signal
    label-free: no gloss, no translation, no gold label of any kind is consulted.
    """
    rng = np.random.default_rng(seed)
    sel = pose_paths if len(pose_paths) <= n_fit else \
        [pose_paths[i] for i in rng.choice(len(pose_paths), n_fit, replace=False)]
    poses, hands = [], []
    for p in sel:
        z = np.load(p, allow_pickle=True)
        poses.append(z["pose"])
        hands.append((z["lh"], z["rh"]))
    feat = ArticulatoryFeaturizer(ArticulatoryConfig(n_handshape_clusters=n_hs, seed=seed))
    feat.fit(poses, hands)
    return feat


def artic_features(feat: ArticulatoryFeaturizer, pose_paths: Sequence[Path]) -> List[Dict[str, float]]:
    out = []
    for p in pose_paths:
        z = np.load(p, allow_pickle=True)
        ok = z["ok"]
        lh = z["lh"].copy(); rh = z["rh"].copy()
        # Zero out frames where the hand was not detected. Feeding undetected hands
        # (written as zeros) into the handshape codebook would mint a spurious
        # "cluster of missing hands" and let detection failure look like articulation.
        lh[~ok[:, 1]] = 0.0
        rh[~ok[:, 2]] = 0.0
        out.append(feat(z["pose"], lh, rh, z["face"]))
    return out


# ---------------------------------------------------------------------------
# selection strategies, all budget-constrained
# ---------------------------------------------------------------------------

def select_random(durations, budget_s, annotation, seed=0, **_):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(durations))
    return _take(order, durations, budget_s, annotation)


def select_duration_random(durations, budget_s, annotation, seed=0, **_):
    """Sample proportional to duration: uniform over annotator-seconds rather than
    over clips. Controls the long-clip confound."""
    rng = np.random.default_rng(seed)
    p = np.asarray(durations, float)
    p = p / p.sum()
    order = rng.choice(len(durations), size=len(durations), replace=False, p=None)
    order = sorted(order, key=lambda i: -p[i] * rng.random())
    return _take(order, durations, budget_s, annotation)


def select_phon_coverage(durations, budget_s, annotation, artic_feats=None,
                         g="sqrt", **_):
    """Cost-effective lazy greedy on submodular articulatory coverage.

    Score is marginal coverage gain per annotator-second, not per clip. Without the
    cost normalization the method degenerates into buying the longest videos, which
    is precisely what duration_random exists to detect.
    """
    import heapq
    cov = ChannelCoverage(g=g)
    heap = []
    for i, f in enumerate(artic_feats):
        c = clip_cost(durations[i], annotation)
        heap.append((-cov.gain(f) / max(c, 1e-9), i, 0))
    heapq.heapify(heap)

    chosen, spent, it = [], 0.0, 0
    while heap:
        neg, i, stamp = heapq.heappop(heap)
        c = clip_cost(durations[i], annotation)
        if spent + c > budget_s:
            continue
        if stamp != it:                      # stale gain, re-score and reinsert
            heapq.heappush(heap, (-cov.gain(artic_feats[i]) / max(c, 1e-9), i, it))
            continue
        chosen.append(i)
        cov.add(artic_feats[i])
        spent += c
        it += 1
    return chosen


def select_phon_coverage_lenmatched(durations, budget_s, annotation, artic_feats=None,
                                    n_bins=10, seed=0, g="sqrt", **_):
    """Coverage greedy constrained to match the pool's clip-length distribution.

    Built to test a hypothesis that turned out to be WRONG. Keep it as the control.

    The hypothesis. Scoring coverage gain per annotator-second makes short clips cheap
    and the greedy exploits this, buying 126 clips of 2.4s at a 0.5h budget where
    uniform random buys 63 of 4.8s. Equal budget means equal total video, so this is
    the same material cut into smaller pieces, and it was plausible that the resulting
    shift away from the evaluation set's length distribution explained the loss.

    Clips are therefore partitioned into duration deciles, each decile gets the budget
    share uniform random would spend there, and the greedy runs within deciles. The
    result has random's length profile and coverage's preferences.

    The outcome. It did not help. Length-matched coverage scored 0.32 / 0.47 / 3.05 /
    3.02 BLEU at 0.5 / 1 / 2 / 5 annotator-hours against unconstrained coverage's
    0.45 / 1.32 / 1.99 / 3.90, so it was worse at three of four budgets. Separately,
    the representativeness-only selector buys the *shortest* clips of any strategy
    (1.5s) and is the best performer. Clip length is a correlate, not the cause.

    What appears to matter instead is typicality: facility location selects clips close
    to many others and beats random, while articulatory coverage selects rare patterns
    and loses. This function stays in the codebase because ruling the length
    explanation out is what licenses that conclusion.
    """
    import heapq
    dur = np.asarray(durations, float)
    edges = np.quantile(dur, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    bin_of = np.clip(np.digitize(dur, edges) - 1, 0, n_bins - 1)

    # Budget share per bin = that bin's share of total pool cost, which is what
    # uniform random sampling spends there in expectation.
    pool_cost = np.array([clip_cost(dur[bin_of == b].sum(), annotation)
                          if (bin_of == b).any() else 0.0 for b in range(n_bins)])
    share = pool_cost / max(pool_cost.sum(), 1e-9)

    cov = ChannelCoverage(g=g)
    chosen: List[int] = []
    # Interleave bins so a shared coverage state is updated fairly rather than one
    # bin claiming all the novel tokens before the others are considered.
    remaining = {b: budget_s * share[b] for b in range(n_bins)}
    heaps = {}
    for b in range(n_bins):
        idx = np.nonzero(bin_of == b)[0]
        h = [(-cov.gain(artic_feats[i]), int(i), 0) for i in idx]
        heapq.heapify(h)
        heaps[b] = h

    it, progress = 0, True
    while progress:
        progress = False
        for b in range(n_bins):
            h = heaps[b]
            while h:
                neg, i, stamp = heapq.heappop(h)
                c = clip_cost(dur[i], annotation)
                if c > remaining[b]:
                    continue
                if stamp != it:
                    heapq.heappush(h, (-cov.gain(artic_feats[i]), i, it))
                    continue
                chosen.append(i)
                cov.add(artic_feats[i])
                remaining[b] -= c
                it += 1
                progress = True
                break
    return chosen


def select_uncertainty(durations, budget_s, annotation, scores=None, **_):
    order = np.argsort(-np.asarray(scores))
    return _take(order, durations, budget_s, annotation)


def _take(order, durations, budget_s, annotation):
    chosen, spent = [], 0.0
    for i in order:
        c = clip_cost(durations[int(i)], annotation)
        if spent + c <= budget_s:
            chosen.append(int(i))
            spent += c
    return chosen


STRATEGIES = {
    "random": select_random,
    "duration_random": select_duration_random,
    "phon_coverage": select_phon_coverage,
    "phon_coverage_lenmatched": select_phon_coverage_lenmatched,
    "uncertainty": select_uncertainty,
}


def coverage_profile(artic_feats: Sequence[Dict[str, float]],
                     indices: Sequence[int]) -> Dict[str, float]:
    """Per-channel inventory coverage of a selected set.

    This is the diagnostic that is far less noisy than BLEU: it says directly how
    much of each articulatory channel the purchase actually touched.
    """
    from collections import defaultdict
    total, got = defaultdict(set), defaultdict(set)
    for f in artic_feats:
        for k in f:
            total[k.split(":")[0]].add(k)
    for i in indices:
        for k in artic_feats[i]:
            got[k.split(":")[0]].add(k)
    out = {f"cov_{ch}": len(got[ch]) / max(len(total[ch]), 1) for ch in total}
    out["cov_all"] = sum(len(got[c]) for c in total) / max(sum(len(total[c]) for c in total), 1)
    return out
