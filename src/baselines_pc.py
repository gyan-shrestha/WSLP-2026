"""The five selection baselines from impl. note 8 that were still missing.

    8.2  source-stratified random   allocate duration evenly over source videos
    8.3  generic pose k-center      farthest-point on the 128-d embedding
    8.6  entropy active learning    2% seed model, then highest frame entropy
    8.7  text facility location     PRIVILEGED, uses hidden translations
    8.8  oracle gloss coverage      PRIVILEGED, uses hidden gloss counts

The last two are diagnostics, not competitors. They answer "how much is left on the
table by not having labels", which is the only way to tell whether a label-free method
is close to the ceiling or nowhere near it. They must never be reported as if they
were label-free, and they are named `privileged_*` here so a results table cannot
quietly mix them in with the rest.

Entropy AL is the one baseline that spends part of the budget on itself: the 2% seed
has to be annotated before any model can score the pool, so per impl. note 8.6 that
seed counts against the budget. Ignoring that would let it compare a 10% subset
against everyone else's 10%, having actually used 12%.
"""
from __future__ import annotations

import heapq
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from acquire import clip_cost


def _take(order, durations, budget_s, annotation) -> List[int]:
    chosen, spent = [], 0.0
    for i in order:
        c = clip_cost(durations[int(i)], annotation)
        if spent + c <= budget_s:
            chosen.append(int(i))
            spent += c
    return chosen


# ---------------------------------------------------------------------------
# 8.2 source-stratified random
# ---------------------------------------------------------------------------

def select_source_random(durations, budget_s, annotation, sources=None,
                         seed=0, **_) -> List[int]:
    """Spread the budget approximately evenly across source videos, sampling randomly
    within each. Controls for the possibility that a coverage method wins merely by
    spreading across recordings rather than by anything articulatory."""
    rng = np.random.default_rng(seed)
    dur = np.asarray(durations, float)
    by_src: Dict[str, List[int]] = {}
    for i, s in enumerate(sources):
        by_src.setdefault(s, []).append(i)

    per_src = budget_s / max(len(by_src), 1)
    chosen, leftover = [], 0.0
    for s, idx in sorted(by_src.items()):
        rng.shuffle(idx)
        allow, spent = per_src + leftover, 0.0
        for i in idx:
            c = clip_cost(dur[i], annotation)
            if spent + c <= allow:
                chosen.append(i)
                spent += c
        leftover = allow - spent          # unspent share rolls to the next source
    return chosen


# ---------------------------------------------------------------------------
# 8.3 generic pose k-center
# ---------------------------------------------------------------------------

def select_kcenter(durations, budget_s, annotation, embeddings=None, **_) -> List[int]:
    """Greedy farthest-point on the generic embedding, with the distance gain divided
    by clip duration (impl. note 8.3)."""
    Z = embeddings
    dur = np.asarray(durations, float)
    n = len(Z)
    mind = np.full(n, np.inf)
    start = int(np.argmax(np.linalg.norm(Z - Z.mean(0), axis=1)))   # deterministic
    chosen, spent = [], 0.0

    cur = start
    while True:
        c = clip_cost(dur[cur], annotation)
        if spent + c > budget_s:
            break
        chosen.append(cur)
        spent += c
        d = np.linalg.norm(Z - Z[cur], axis=1)
        mind = np.minimum(mind, d)
        mind[chosen] = -np.inf
        score = mind / np.maximum(dur * clip_cost(1.0, annotation), 1e-9)
        cur = int(np.argmax(score))
        if not np.isfinite(mind[cur]) or mind[cur] <= 0:
            break
    return chosen


# ---------------------------------------------------------------------------
# 8.6 entropy active learning
# ---------------------------------------------------------------------------

def select_entropy(durations, budget_s, annotation, entropy_scores=None,
                   seed_frac: float = 0.02, seed: int = 0, **_) -> List[int]:
    """Random 2% seed, then highest length-normalised entropy until the budget is met.

    The seed counts against the budget (impl. note 8.6). `entropy_scores` must come
    from a model trained ONLY on the seed, otherwise this stops being active learning
    and becomes an oracle.
    """
    rng = np.random.default_rng(seed)
    dur = np.asarray(durations, float)
    seed_budget = seed_frac * budget_s

    order = rng.permutation(len(dur))
    seed_idx, spent = [], 0.0
    for i in order:
        c = clip_cost(dur[i], annotation)
        if spent + c <= seed_budget:
            seed_idx.append(int(i))
            spent += c

    if entropy_scores is None:                    # seed only, model not yet trained
        return seed_idx

    rest = [i for i in np.argsort(-np.asarray(entropy_scores)) if int(i) not in set(seed_idx)]
    for i in rest:
        c = clip_cost(dur[int(i)], annotation)
        if spent + c <= budget_s:
            seed_idx.append(int(i))
            spent += c
    return seed_idx


# ---------------------------------------------------------------------------
# 8.7 / 8.8 privileged diagnostics
# ---------------------------------------------------------------------------

def privileged_text_facility_location(durations, budget_s, annotation,
                                      translations=None, **_) -> List[int]:
    """PRIVILEGED. TF-IDF facility location over hidden spoken-language translations.

    Upper bound representing access to translations at selection time. Not a
    label-free competitor.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(translations)
    X = np.asarray(X.todense(), dtype=np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    dur = np.asarray(durations, float)

    best = np.zeros(len(X), np.float32)
    heap = []
    for i in range(len(X)):
        g = float(np.maximum(np.maximum(X @ X[i], 0) - best, 0).sum() / len(X))
        heap.append((-g / max(clip_cost(dur[i], annotation), 1e-9), i, 0))
    heapq.heapify(heap)

    chosen, spent, it = [], 0.0, 0
    while heap:
        _, i, stamp = heapq.heappop(heap)
        c = clip_cost(dur[i], annotation)
        if spent + c > budget_s:
            continue
        g = float(np.maximum(np.maximum(X @ X[i], 0) - best, 0).sum() / len(X))
        if stamp != it:
            heapq.heappush(heap, (-g / max(c, 1e-9), i, it))
            continue
        chosen.append(i)
        best = np.maximum(best, np.maximum(X @ X[i], 0))
        spent += c
        it += 1
    return chosen


def privileged_oracle_gloss(durations, budget_s, annotation, glosses=None, **_) -> List[int]:
    """PRIVILEGED. Diminishing-return coverage over hidden gloss counts.

    The ceiling: selection with perfect knowledge of the labels being bought. Any
    label-free method is judged by how much of this gap it closes.
    """
    dur = np.asarray(durations, float)
    counts = [{} for _ in glosses]
    for i, g in enumerate(glosses):
        for t in g:
            counts[i][t] = counts[i].get(t, 0) + 1

    acc: Dict[str, float] = {}

    def gain(i):
        return sum(np.log1p(acc.get(t, 0.0) + v) - np.log1p(acc.get(t, 0.0))
                   for t, v in counts[i].items())

    heap = [(-gain(i) / max(clip_cost(dur[i], annotation), 1e-9), i, 0)
            for i in range(len(dur))]
    heapq.heapify(heap)

    chosen, spent, it = [], 0.0, 0
    while heap:
        _, i, stamp = heapq.heappop(heap)
        c = clip_cost(dur[i], annotation)
        if spent + c > budget_s:
            continue
        if stamp != it:
            heapq.heappush(heap, (-gain(i) / max(c, 1e-9), i, it))
            continue
        chosen.append(i)
        for t, v in counts[i].items():
            acc[t] = acc.get(t, 0.0) + v
        spent += c
        it += 1
    return chosen


PC_BASELINES: Dict[str, Callable] = {
    "source_random": select_source_random,
    "kcenter": select_kcenter,
    "entropy_al": select_entropy,
    "privileged_text_fl": privileged_text_facility_location,
    "privileged_oracle_gloss": privileged_oracle_gloss,
}

# Names that must be reported separately from label-free methods.
PRIVILEGED = {"privileged_text_fl", "privileged_oracle_gloss"}
