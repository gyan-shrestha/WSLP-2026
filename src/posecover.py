"""PoseCover: the full objective and its ablations.

Implements the selector specified in the PoseCover draft, on top of the articulatory
descriptors already built in descriptors.py:

    F(S) = l_art * F_art(S) + l_rep * F_rep(S) + l_src * F_src(S)

with each component normalised by its value on the complete pool, selected by a
duration-constrained lazy greedy scoring marginal gain per annotator-second.

Why the ablation order here differs from the draft's. The draft runs the full method
first and ablations afterwards. We already have the articulatory-only arm, and it
loses to uniform random at every budget by a wide margin. That inverts the useful
question: it is no longer "does the full method win" but "does the articulatory term
contribute anything the representativeness term does not already provide". Running
representativeness-only *before* the full method answers that in one run. If generic
facility location alone matches the full method, the draft's own acceptance criteria
call the result negative, and it is better to know that at the start of the matrix
than at the end.

Components below follow the implementation note: per-clip codeword caps to stop a
long or repetitive clip satisfying coverage by repetition, inverse-frequency rare
weighting with a cap, a quality score that discounts tracking failure without
excluding unconventional signing, and log-diminishing returns on coverage.
"""
from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from acquire import clip_cost


# ---------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------

def quality_score(ok: np.ndarray) -> float:
    """q_i = clip((v + a + r)/3, 0.1, 1.0) from the implementation note.

    v: valid-frame ratio (we use hand validity, the binding constraint)
    a: mean landmark availability across all four streams
    r: temporal continuity, the fraction of adjacent frame pairs whose validity
       does not flip; jittery tracking is penalised even when the mean rate is fine.

    Floored at 0.1 rather than zero so that low-quality clips are down-weighted but
    never made ineligible. Excluding them outright would silently remove the signers
    and recording conditions that track worst, which is precisely the population a
    corpus is trying to represent.
    """
    if ok.size == 0:
        return 0.1
    hands = ok[:, 1:3].any(axis=1)
    v = float(hands.mean())
    a = float(ok.mean())
    r = 1.0 - float(np.abs(np.diff(hands.astype(float))).mean()) if len(hands) > 1 else 1.0
    return float(np.clip((v + a + r) / 3.0, 0.1, 1.0))


# ---------------------------------------------------------------------------
# generic pose embedding, for the representativeness term
# ---------------------------------------------------------------------------

def generic_embeddings(pose_paths: Sequence[Path], n_frames: int = 64,
                       n_pca: int = 128, seed: int = 0) -> np.ndarray:
    """Resample -> flatten -> standardise -> PCA -> L2 normalise.

    PCA is fit on the unlabelled pool only. This representation deliberately does not
    know about articulatory structure: it is the "generic" view whose whole purpose is
    to be a control on whether sign-specific descriptors add anything.
    """
    from sklearn.decomposition import PCA

    rows = []
    for p in pose_paths:
        z = np.load(p, allow_pickle=True)
        pose, lh, rh, ok = z["pose"], z["lh"], z["rh"], z["ok"]
        T = len(pose)
        idx = np.linspace(0, max(T - 1, 0), n_frames).astype(int) if T else np.zeros(n_frames, int)
        body = pose[idx][:, :25, :2].astype(np.float32)
        mid = (body[:, 11] + body[:, 12]) / 2.0
        w = np.maximum(np.linalg.norm(body[:, 11] - body[:, 12], axis=-1), 1e-3)[:, None]
        nb = (body - mid[:, None, :]) / w[:, None, :]
        nl = (lh[idx][:, :, :2].astype(np.float32) - mid[:, None, :]) / w[:, None, :]
        nr = (rh[idx][:, :, :2].astype(np.float32) - mid[:, None, :]) / w[:, None, :]
        d = np.diff(nb, axis=0, prepend=nb[:1])
        feat = np.concatenate([nb.reshape(n_frames, -1), nl.reshape(n_frames, -1),
                               nr.reshape(n_frames, -1), d.reshape(n_frames, -1),
                               ok[idx].astype(np.float32)], axis=1)
        rows.append(np.nan_to_num(feat, nan=0.0).ravel())

    X = np.asarray(rows, dtype=np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Z = PCA(n_components=min(n_pca, X.shape[0] - 1, X.shape[1]),
            random_state=seed).fit_transform(X)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------------------
# the objective
# ---------------------------------------------------------------------------

@dataclass
class PoseCoverConfig:
    w_art: float = 0.60
    w_rep: float = 0.30
    w_src: float = 0.10
    per_clip_cap: int = 8        # caps n_igk; stops repetition satisfying coverage
    rare_weight_cap: float = 5.0
    use_quality: bool = True
    use_rare_weights: bool = True


class PoseCoverObjective:
    """Monotone submodular, as the weighted sum of three monotone submodular terms.

    Each term is normalised by its value on the complete pool so that the weights mean
    the same thing regardless of the corpus, which is what makes 0.60/0.30/0.10
    transferable rather than a PHOENIX14T-specific tuning.
    """

    def __init__(self, cfg: PoseCoverConfig,
                 artic_feats: Sequence[Dict[str, float]],
                 embeddings: Optional[np.ndarray],
                 quality: Optional[np.ndarray],
                 sources: Optional[Sequence[str]]):
        self.cfg = cfg
        self.q = (np.asarray(quality, float) if (quality is not None and cfg.use_quality)
                  else np.ones(len(artic_feats)))

        # capped per-clip counts
        self.counts: List[Dict[str, float]] = [
            {k: min(v, cfg.per_clip_cap) for k, v in f.items()} for f in artic_feats]

        # pool frequency and inverse-frequency rare weights, per channel
        pool: Dict[str, float] = {}
        for c in self.counts:
            for k, v in c.items():
                pool[k] = pool.get(k, 0.0) + v
        self.rare: Dict[str, float] = {}
        by_ch: Dict[str, List[float]] = {}
        for k, v in pool.items():
            by_ch.setdefault(k.split(":")[0], []).append(v)
        med = {ch: float(np.median(v)) for ch, v in by_ch.items()}
        for k, v in pool.items():
            if cfg.use_rare_weights:
                self.rare[k] = min(cfg.rare_weight_cap,
                                   float(np.sqrt((med[k.split(':')[0]] + 1e-9) / (v + 1e-9))))
            else:
                self.rare[k] = 1.0

        self.channels = sorted(by_ch)
        self.acc: Dict[str, float] = {}

        # representativeness: cosine similarity to every pool member
        self.emb = embeddings
        self.best = np.zeros(len(artic_feats)) if embeddings is not None else None

        self.sources = list(sources) if sources is not None else None
        if self.sources is not None:
            self.src_pool: Dict[str, int] = {}
            for s in self.sources:
                self.src_pool[s] = self.src_pool.get(s, 0) + 1
            self.src_sel: Dict[str, int] = {s: 0 for s in self.src_pool}

        self._norm = {"art": 1.0, "rep": 1.0, "src": 1.0}
        self._norm = self._pool_values()

    # -- component values on a hypothetical addition ------------------------
    def _art_gain(self, i: int) -> float:
        g = 0.0
        for ch in self.channels:
            s = 0.0
            n = 0
            for k, v in self.counts[i].items():
                if k.split(":")[0] != ch:
                    continue
                n += 1
                a = self.acc.get(k, 0.0)
                s += self.rare[k] * (np.log1p(a + self.q[i] * v) - np.log1p(a))
            if n:
                g += s / max(len(self.channels), 1)
        return g

    def _rep_gain(self, i: int) -> float:
        if self.emb is None:
            return 0.0
        sim = np.maximum(self.emb @ self.emb[i], 0.0)
        return float(np.maximum(sim - self.best, 0.0).sum() / len(self.emb))

    def _src_gain(self, i: int) -> float:
        if self.sources is None:
            return 0.0
        s = self.sources[i]
        before = np.log1p(self.src_sel[s]) / np.log1p(self.src_pool[s])
        after = np.log1p(self.src_sel[s] + 1) / np.log1p(self.src_pool[s])
        return float((after - before) / len(self.src_pool))

    def _pool_values(self) -> Dict[str, float]:
        """Component values when the entire pool is selected, used for normalisation."""
        art = 0.0
        for ch in self.channels:
            tot: Dict[str, float] = {}
            for i, c in enumerate(self.counts):
                for k, v in c.items():
                    if k.split(":")[0] == ch:
                        tot[k] = tot.get(k, 0.0) + self.q[i] * v
            art += sum(self.rare[k] * np.log1p(v) for k, v in tot.items()) / max(len(self.channels), 1)
        rep = 1.0 if self.emb is not None else 1.0
        src = 1.0 if self.sources is not None else 1.0
        return {"art": max(art, 1e-9), "rep": max(rep, 1e-9), "src": max(src, 1e-9)}

    def gain(self, i: int) -> float:
        c = self.cfg
        return (c.w_art * self._art_gain(i) / self._norm["art"]
                + c.w_rep * self._rep_gain(i) / self._norm["rep"]
                + c.w_src * self._src_gain(i) / self._norm["src"])

    def add(self, i: int) -> None:
        for k, v in self.counts[i].items():
            self.acc[k] = self.acc.get(k, 0.0) + self.q[i] * v
        if self.emb is not None:
            self.best = np.maximum(self.best, np.maximum(self.emb @ self.emb[i], 0.0))
        if self.sources is not None:
            self.src_sel[self.sources[i]] += 1


def select_posecover(durations, budget_s, annotation, artic_feats=None,
                     embeddings=None, quality=None, sources=None,
                     cfg: PoseCoverConfig = PoseCoverConfig(), **_) -> List[int]:
    """Duration-constrained lazy greedy on the full objective."""
    obj = PoseCoverObjective(cfg, artic_feats, embeddings, quality, sources)
    dur = np.asarray(durations, float)

    heap = [(-obj.gain(i) / max(clip_cost(dur[i], annotation), 1e-9), int(i), 0)
            for i in range(len(dur))]
    heapq.heapify(heap)

    chosen: List[int] = []
    spent, it = 0.0, 0
    while heap:
        neg, i, stamp = heapq.heappop(heap)
        c = clip_cost(dur[i], annotation)
        if spent + c > budget_s:
            continue
        if stamp != it:
            heapq.heappush(heap, (-obj.gain(i) / max(c, 1e-9), i, it))
            continue
        chosen.append(i)
        obj.add(i)
        spent += c
        it += 1
    return chosen


# ---------------------------------------------------------------------------
# ablation presets, in the order that answers the live question first
# ---------------------------------------------------------------------------

ABLATIONS = {
    # does generic representativeness alone suffice? if yes, the articulatory term
    # is not earning its place and the draft's own criteria call that negative
    "rep_only":      PoseCoverConfig(w_art=0.0,  w_rep=1.0,  w_src=0.0),
    "art_only":      PoseCoverConfig(w_art=1.0,  w_rep=0.0,  w_src=0.0),
    "art_rep":       PoseCoverConfig(w_art=0.67, w_rep=0.33, w_src=0.0),
    "full":          PoseCoverConfig(w_art=0.60, w_rep=0.30, w_src=0.10),
    "full_no_rare":  PoseCoverConfig(use_rare_weights=False),
    "full_no_qual":  PoseCoverConfig(use_quality=False),
}
