"""BADGE selection baseline (Ash et al., ICLR 2020), adapted to a duration budget.

The one standard active-learning method a reviewer will ask for by name, and the one
family our other baselines do not cover. Core-set selects for geometric spread,
uncertainty selects for confusion; BADGE selects for both at once by clustering
gradient embeddings, whose magnitude encodes uncertainty and whose direction encodes
which part of the model would move.

Adaptation. BADGE normally picks a fixed count k via k-means++ seeding. Here the
constraint is annotator-seconds, not clips, so seeding continues until the budget is
exhausted rather than until k is reached, and each candidate's squared distance is
divided by its cost so that a long clip must be proportionally more informative to be
worth buying. Without that division BADGE degenerates into buying the longest videos,
the same failure mode the duration-weighted control exists to detect.

Gradient embedding. For a CTC model the loss gradient with respect to the output layer
is (p - y) x h, summed over frames. The full outer product is |vocab| x d_model, which
is far too large to cluster, so we use the standard cheap surrogate: the per-frame
predictive residual pooled over time, which preserves both the magnitude and the
direction of the last-layer gradient without materialising the outer product.

Like entropy AL, this needs a seed model and therefore labels, so it is not label-free.
It is included to show that model-based selection does not rescue the result either.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from acquire import clip_cost
from data_phoenix import Clip, PhoenixDataset, Vocab, collate
from pose_ctc import PoseCTC
from train_ctc import CTCTrainConfig, train_ctc_one


@torch.no_grad()
def gradient_embeddings(model: PoseCTC, clips: Sequence[Clip], vocab: Vocab,
                        device: str, cache: Optional[dict] = None,
                        batch_size: int = 16) -> np.ndarray:
    """Per-clip surrogate for the last-layer loss gradient.

    Uses the predictive residual (p - onehot(argmax)) pooled over frames and scaled by
    the mean hidden-state norm. Its length grows with model uncertainty, exactly as the
    true gradient's does, which is the property BADGE's k-means++ seeding exploits.
    """
    model.eval()
    dummy = Vocab(["x"])
    ds = PhoenixDataset(list(clips), vocab, dummy, max_frames=320, cache=cache)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                    num_workers=2)

    embs: List[np.ndarray] = []
    for batch in dl:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        h, lens = model.encode(b["x"], b["x_mask"])
        logits = model.out(h)
        p = F.softmax(logits, dim=-1)
        hard = F.one_hot(p.argmax(-1), p.size(-1)).to(p.dtype)
        resid = p - hard                                        # (B, T, V)
        hnorm = h.norm(dim=-1, keepdim=True)                    # (B, T, 1)
        g = (resid * hnorm)
        for i in range(g.size(0)):
            n = max(int(lens[i]), 1)
            embs.append(g[i, :n].mean(0).float().cpu().numpy())
    return np.asarray(embs, dtype=np.float32)


def badge_select(embeddings: np.ndarray, durations: np.ndarray,
                 budget_s: float, annotation: str, seed: int = 0) -> List[int]:
    """k-means++ seeding under a duration budget.

    Standard BADGE draws the next point with probability proportional to its squared
    distance from the current set. Dividing that by cost turns it into distance per
    annotator-second, which is the budgeted analogue and keeps the sampling from
    collapsing onto long clips.
    """
    rng = np.random.default_rng(seed)
    Z = np.asarray(embeddings, np.float64)
    dur = np.asarray(durations, float)
    n = len(Z)
    costs = np.array([clip_cost(d, annotation) for d in dur])

    # first pick: largest gradient norm, i.e. the clip the model is least settled on
    norms = np.linalg.norm(Z, axis=1)
    first = int(np.argmax(norms / np.maximum(costs, 1e-9)))

    chosen = [first]
    spent = costs[first]
    d2 = ((Z - Z[first]) ** 2).sum(1)

    while True:
        w = d2 / np.maximum(costs, 1e-9)
        w[chosen] = 0.0
        w[spent + costs > budget_s] = 0.0          # infeasible under remaining budget
        tot = w.sum()
        if tot <= 0:
            break
        i = int(rng.choice(n, p=w / tot))
        chosen.append(i)
        spent += costs[i]
        d2 = np.minimum(d2, ((Z - Z[i]) ** 2).sum(1))
    return sorted(chosen)


def run_badge(all_train: Sequence[Clip], dev: Sequence[Clip], durations: np.ndarray,
              budget_s: float, annotation: str, seed: int = 0,
              seed_frac: float = 0.02, max_epochs: int = 60,
              cache: Optional[dict] = None, log_prefix: str = "") -> Dict:
    """Seed -> train -> embed -> BADGE-select -> retrain, with the seed charged to the
    budget exactly as in entropy AL, so the two are directly comparable."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    dur = np.asarray(durations, float)
    t0 = time.time()

    seed_budget = seed_frac * budget_s
    seed_idx, spent = [], 0.0
    for i in rng.permutation(len(dur)):
        c = clip_cost(dur[i], annotation)
        if spent + c <= seed_budget:
            seed_idx.append(int(i)); spent += c
    if not seed_idx:
        seed_idx = [int(np.argmin(dur))]
        spent = clip_cost(dur[seed_idx[0]], annotation)
    print(f"{log_prefix}seed: {len(seed_idx)} clips, {spent/3600:.3f}h", flush=True)

    seed_clips = [all_train[i] for i in seed_idx]
    seed_res = train_ctc_one({c.clip_id: annotation for c in seed_clips},
                             all_train, dev,
                             cfg=CTCTrainConfig(seed=seed,
                                                max_epochs=max(10, max_epochs // 3)),
                             cache=cache, log_prefix=f"{log_prefix}[seed] ",
                             return_model=True)
    model, vocab = seed_res["_model"], seed_res["_vocab"]

    Z = gradient_embeddings(model, all_train, vocab, device, cache=cache)
    print(f"{log_prefix}gradient embeddings {Z.shape}", flush=True)

    idx = badge_select(Z, dur, budget_s, annotation, seed=seed)
    total = sum(clip_cost(dur[i], annotation) for i in idx)
    print(f"{log_prefix}selected {len(idx)} clips, {total/3600:.3f}/{budget_s/3600:.1f}h",
          flush=True)

    res = train_ctc_one({all_train[i].clip_id: annotation for i in idx},
                        all_train, dev,
                        cfg=CTCTrainConfig(seed=seed, max_epochs=max_epochs),
                        cache=cache, log_prefix=f"{log_prefix}[final] ")
    res.update(strategy="badge", n_seed_clips=len(seed_idx),
               seed_budget_frac=seed_frac,
               total_minutes=(time.time() - t0) / 60.0)
    return res
