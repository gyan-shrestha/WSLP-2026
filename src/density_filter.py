"""Does removing atypical candidates rescue coverage selection?

Part two of the "why does coverage fail" experiment. If the density analysis shows that
articulatory coverage buys clips from sparse regions of the pool, the natural follow-up
is to take those clips off the menu and rerun the identical selector:

  1. compute local density over the unlabeled pool
  2. discard the least dense fraction of candidates
  3. run exactly the same coverage selection on what remains

Two outcomes, both worth reporting.

  Performance improves. Naive coverage maximization was confounded by atypical
  samples, and a representativeness constraint partially rescues it.

  Performance still trails random. Controlling for outliers does not rescue coverage,
  which is stronger evidence that articulatory diversity is a poor proxy for annotation
  utility rather than a good idea spoiled by noise.

One caveat belongs in any writeup of the positive case. Filtering by density in the
generic embedding *is* a representativeness constraint, applied as a hard pre-filter
rather than as a term in the objective. So an improvement should be read as "coverage
needs a representativeness constraint to be usable", not as "coverage works once the
data is cleaned". The filter is doing the work that the coverage objective was supposed
to do on its own.

Density is computed on the pool before any selection, using no labels, so the filtered
selector remains label-free and comparable to the others.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from density import knn_distance
from posecover import PoseCoverConfig, select_posecover


def density_mask(embeddings: np.ndarray, drop_frac: float = 0.15,
                 k: int = 20) -> np.ndarray:
    """Boolean keep-mask retaining all but the sparsest `drop_frac` of the pool.

    Sparsity is mean distance to the k nearest neighbours, so the discarded clips are
    those least like anything else in the pool.
    """
    d = knn_distance(embeddings, k=k)
    cutoff = np.percentile(d, 100.0 * (1.0 - drop_frac))
    return d < cutoff


def select_density_filtered(durations, budget_s, annotation, artic_feats=None,
                            embeddings=None, quality=None, sources=None,
                            cfg: Optional[PoseCoverConfig] = None,
                            drop_frac: float = 0.15, k: int = 20,
                            precomputed_mask: Optional[np.ndarray] = None,
                            **_) -> List[int]:
    """Identical PoseCover selection, run on a density-filtered candidate pool.

    Everything except the candidate set is unchanged: same objective, same weights,
    same greedy, same budget. That is what makes the comparison interpretable, since
    any difference is attributable to which clips were eligible rather than to how they
    were scored.
    """
    dur = np.asarray(durations, float)
    keep = precomputed_mask if precomputed_mask is not None else \
        density_mask(embeddings, drop_frac=drop_frac, k=k)
    idx_map = np.nonzero(keep)[0]

    # Re-index every pool-level input to the surviving candidates. The embedding is
    # subset too, so representativeness is computed against the filtered pool; a
    # selector cannot be asked to represent clips that are no longer purchasable.
    sub_feats = [artic_feats[i] for i in idx_map]
    sub_emb = embeddings[idx_map] if embeddings is not None else None
    sub_qual = np.asarray(quality)[idx_map] if quality is not None else None
    sub_srcs = [sources[i] for i in idx_map] if sources is not None else None

    local = select_posecover(dur[idx_map], budget_s, annotation,
                             artic_feats=sub_feats, embeddings=sub_emb,
                             quality=sub_qual, sources=sub_srcs,
                             cfg=cfg or PoseCoverConfig())
    return sorted(int(idx_map[i]) for i in local)


# Configurations mirroring the unfiltered ablations, so each filtered run has an exact
# counterpart in the existing results.
FILTERED = {
    "art_only": PoseCoverConfig(w_art=1.0, w_rep=0.0, w_src=0.0),
    "full":     PoseCoverConfig(w_art=0.60, w_rep=0.30, w_src=0.10),
}

# Both ends of the specified 10 to 20 percent range. Reporting one value cannot show
# whether the conclusion depends on where the threshold is drawn, and a result that
# flips between 10 and 20 percent means something different from one that holds at both.
DROP_FRACTIONS = (0.10, 0.20)

# Strategy names understood by run_budget.py, of the form:
#     dens{drop_percent}_{config}      e.g. dens10_art_only, dens20_full
STRATEGY_NAMES = [f"dens{int(d*100)}_{c}" for d in DROP_FRACTIONS for c in FILTERED]


def parse_strategy(name: str):
    """'dens10_art_only' -> (0.10, PoseCoverConfig for art_only). None if not ours."""
    if not name.startswith("dens"):
        return None
    try:
        head, cfg_name = name.split("_", 1)
        drop = int(head[4:]) / 100.0
    except (ValueError, IndexError):
        return None
    if cfg_name not in FILTERED:
        return None
    return drop, FILTERED[cfg_name]
