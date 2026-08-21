"""Evaluation beyond aggregate WER (paper 6.2-6.7, impl. note 13).

Aggregate WER hides the thing a selection method is actually supposed to affect. A
strategy can lower WER simply by buying clips that are easy to recognise, without ever
broadening what the model can recognise at all. These metrics separate the two:

    AULC             quality per unit of annotation, across the whole budget curve
    OOV rate         dev tokens the purchase never bought, unreachable by construction
    rare-gloss recall  whether uncommon lexical material was acquired at all
    gloss coverage   fraction of the pool vocabulary the purchase contains
    source entropy   whether the budget concentrated on a few recordings
    redundancy       whether the selected clips merely resemble each other
    bootstrap CI     paired, so two systems are compared on the same utterances
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from pose_ctc import levenshtein


# ---------------------------------------------------------------------------
# 6.2 annotation-efficiency curve
# ---------------------------------------------------------------------------

def aulc(budgets: Sequence[float], wers: Sequence[float]) -> float:
    """Normalised area under the annotation learning curve (paper Eq. 30).

    Accuracy is 1 - WER, integrated by the trapezoid rule over budget and divided by
    the largest budget.

    Eq. 30 integrates from the FIRST budget to the last and divides by b_max, so a
    perfect system scores 1 only when the grid starts at 0. On a grid of [1, 2, 3] a
    perfect system scores 2/3. That is a property of the definition rather than a bug,
    but it means AULC is comparable only between runs sharing a budget grid.
    """
    b = np.asarray(budgets, float)
    a = 1.0 - np.asarray(wers, float) / 100.0
    o = np.argsort(b)
    b, a = b[o], a[o]
    if len(b) < 2:
        return float(a[0]) if len(a) else 0.0
    trapezoids = (a[:-1] + a[1:]) / 2.0 * np.diff(b)
    return float(trapezoids.sum() / b[-1])


# ---------------------------------------------------------------------------
# 6.3 vocabulary coverage
# ---------------------------------------------------------------------------

def vocab_metrics(selected_glosses: Sequence[Sequence[str]],
                  pool_glosses: Sequence[Sequence[str]],
                  dev_glosses: Sequence[Sequence[str]],
                  rare_max: int = 5) -> Dict[str, float]:
    """Gloss-type coverage, rare-type coverage, and dev OOV token rate (Eq. 31-34)."""
    sel = {t for g in selected_glosses for t in g}
    pool_counts = Counter(t for g in pool_glosses for t in g)
    pool_types = set(pool_counts)
    rare = {t for t, n in pool_counts.items() if n <= rare_max}
    singleton = {t for t, n in pool_counts.items() if n == 1}

    dev_tok = [t for g in dev_glosses for t in g]
    oov = sum(t not in sel for t in dev_tok) / max(len(dev_tok), 1)

    return {
        "gloss_coverage": len(sel & pool_types) / max(len(pool_types), 1),
        "rare_coverage": len(sel & rare) / max(len(rare), 1),
        "singleton_coverage": len(sel & singleton) / max(len(singleton), 1),
        "dev_oov_token_rate": 100.0 * oov,
        "n_selected_types": len(sel),
    }


# ---------------------------------------------------------------------------
# 6.4 rare-gloss recall
# ---------------------------------------------------------------------------

def rare_gloss_recall(refs: Sequence[Sequence[str]], hyps: Sequence[Sequence[str]],
                      pool_glosses: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Recall by pool frequency band: 1-5, 6-20, >20 occurrences (impl. note 13.2).

    Frequency bands come from the FULL pool, which the selector never sees. They are an
    analysis device applied after selection, not a signal available during it.
    """
    pool = Counter(t for g in pool_glosses for t in g)
    bands = {"1-5": (1, 5), "6-20": (6, 20), ">20": (21, 10 ** 9)}
    correct = defaultdict(int)
    total = defaultdict(int)

    for r, h in zip(refs, hyps):
        hc = Counter(h)
        for t, n in Counter(r).items():
            f = pool.get(t, 0)
            band = next((b for b, (lo, hi) in bands.items() if lo <= f <= hi), None)
            if band is None:
                continue
            total[band] += n
            correct[band] += min(n, hc.get(t, 0))

    return {f"recall_{b}": (correct[b] / total[b] if total[b] else float("nan"))
            for b in bands} | {f"n_{b}": total[b] for b in bands}


# ---------------------------------------------------------------------------
# 6.6 / 6.7 selection-set diagnostics
# ---------------------------------------------------------------------------

def source_entropy(sources: Sequence[str]) -> float:
    """Normalised entropy over source videos (Eq. 37). Low values mean the budget
    concentrated on a few recordings, which can raise aggregate scores while making
    the system worse for everyone not in them."""
    c = Counter(sources)
    if len(c) <= 1:
        return 0.0
    p = np.array(list(c.values()), float)
    p /= p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(c)))


def redundancy(embeddings: np.ndarray, idx: Sequence[int]) -> float:
    """Mean nearest-neighbour cosine similarity inside the selected set (Eq. 39).

    High redundancy with good WER means the strategy is buying near-duplicates that
    happen to match the evaluation distribution, which is worth knowing.
    """
    if len(idx) < 2:
        return 0.0
    Z = embeddings[list(idx)]
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    return float(S.max(1).mean())


# ---------------------------------------------------------------------------
# 13.3 paired bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap_wer(refs, hyps_a, hyps_b, n_boot: int = 1000, seed: int = 0):
    """Paired bootstrap over test utterances (impl. note 13.3).

    Both systems are resampled on the SAME utterance indices, so the comparison is not
    contaminated by which sentences happened to be drawn.
    """
    rng = np.random.default_rng(seed)
    n = len(refs)
    per = []
    for r, a, b in zip(refs, hyps_a, hyps_b):
        sa, da, ia = levenshtein(r, a)
        sb, db, ib = levenshtein(r, b)
        per.append((sa + da + ia, sb + db + ib, len(r)))
    per = np.asarray(per, float)

    diffs = []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        N = max(per[s, 2].sum(), 1)
        diffs.append(100.0 * (per[s, 0].sum() - per[s, 1].sum()) / N)
    d = np.asarray(diffs)
    N = max(per[:, 2].sum(), 1)
    return {
        "delta_wer": 100.0 * (per[:, 0].sum() - per[:, 1].sum()) / N,
        "ci_low": float(np.percentile(d, 2.5)),
        "ci_high": float(np.percentile(d, 97.5)),
        "p_a_better": float((d < 0).mean()),
    }
