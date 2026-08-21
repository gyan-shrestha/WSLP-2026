"""Entropy-based active learning baseline (impl. note 8.6).

The only baseline here that needs two passes, and the only one that spends part of its
budget on itself:

    1. randomly select a 2% seed
    2. train the CTC model on the seed
    3. score every unselected clip by length-normalised frame entropy
    4. add clips by descending entropy until the target budget is reached
    5. retrain from scratch on the complete selection

The seed counts against the budget. Skipping that would let this method compare a
nominal 10% subset against everyone else's 10% while having actually consumed 12%,
which is the most common way active-learning baselines are quietly favoured.

Length normalisation matters for the same reason duration normalisation matters in the
coverage selectors: total frame entropy grows with clip length, so an unnormalised
score is a long-clip detector wearing an uncertainty costume.

Note the asymmetry this baseline carries by construction. It is the only strategy that
needs labels before it can select, which is precisely the situation a label-free method
is meant to avoid. It is included because the spec asks for it and because it bounds
how much model-derived uncertainty is worth here, not as a like-for-like competitor.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from acquire import clip_cost
from data_phoenix import Clip, PhoenixDataset, Vocab, collate, FEATURE_DIM
from pose_ctc import CTCConfig, PoseCTC
from train_ctc import CTCTrainConfig, build_subset_vocab, train_ctc_one


@torch.no_grad()
def frame_entropy_scores(model: PoseCTC, clips: Sequence[Clip], vocab: Vocab,
                         device: str, cache: Optional[dict] = None,
                         batch_size: int = 16) -> np.ndarray:
    """Mean per-frame predictive entropy over the encoded sequence.

    Averaged rather than summed, so the score reflects how uncertain the model is per
    unit of signing rather than how much signing there is.
    """
    model.eval()
    dummy = Vocab(["x"])
    ds = PhoenixDataset(list(clips), vocab, dummy, max_frames=320, cache=cache)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                    num_workers=2)
    out: List[float] = []
    for batch in dl:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        h, lens = model.encode(b["x"], b["x_mask"])
        logp = F.log_softmax(model.out(h), dim=-1)
        ent = -(logp.exp() * logp).sum(-1)                      # (B, T)
        for i in range(ent.size(0)):
            n = max(int(lens[i]), 1)
            out.append(float(ent[i, :n].mean()))
    return np.asarray(out, dtype=np.float64)


def run_entropy_al(all_train: Sequence[Clip], dev: Sequence[Clip],
                   durations: np.ndarray, budget_s: float, annotation: str,
                   seed: int = 0, seed_frac: float = 0.02,
                   max_epochs: int = 60, cache: Optional[dict] = None,
                   log_prefix: str = "") -> Dict:
    """Full two-pass procedure. Returns the trained result plus selection provenance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    dur = np.asarray(durations, float)
    t0 = time.time()

    # --- 1. random 2% seed, charged to the budget --------------------------
    seed_budget = seed_frac * budget_s
    order = rng.permutation(len(dur))
    seed_idx: List[int] = []
    spent = 0.0
    for i in order:
        c = clip_cost(dur[i], annotation)
        if spent + c <= seed_budget:
            seed_idx.append(int(i))
            spent += c
    if not seed_idx:                       # tiny budgets: take the single cheapest clip
        seed_idx = [int(np.argmin(dur))]
        spent = clip_cost(dur[seed_idx[0]], annotation)
    print(f"{log_prefix}seed: {len(seed_idx)} clips, {spent/3600:.3f}h "
          f"({100*spent/budget_s:.1f}% of budget)", flush=True)

    # --- 2. train the seed model -------------------------------------------
    seed_clips = [all_train[i] for i in seed_idx]
    seed_res = train_ctc_one({c.clip_id: annotation for c in seed_clips},
                             all_train, dev,
                             cfg=CTCTrainConfig(seed=seed,
                                                max_epochs=max(10, max_epochs // 3)),
                             cache=cache, log_prefix=f"{log_prefix}[seed] ",
                             return_model=True)
    model, seed_vocab = seed_res["_model"], seed_res["_vocab"]
    print(f"{log_prefix}seed model: vocab={len(seed_vocab)} "
          f"dev WER={seed_res['dev']['wer']:.1f}", flush=True)

    # --- 3. score the unselected pool --------------------------------------
    chosen = set(seed_idx)
    rest_idx = [i for i in range(len(dur)) if i not in chosen]
    scores = frame_entropy_scores(model, [all_train[i] for i in rest_idx],
                                  seed_vocab, device, cache=cache)

    # --- 4. fill the remaining budget by descending entropy ----------------
    for pos in np.argsort(-scores):
        i = rest_idx[int(pos)]
        c = clip_cost(dur[i], annotation)
        if spent + c <= budget_s:
            chosen.add(i)
            spent += c
    final = sorted(chosen)
    print(f"{log_prefix}final: {len(final)} clips, {spent/3600:.3f}/{budget_s/3600:.1f}h",
          flush=True)

    # --- 5. retrain from scratch on the full selection ---------------------
    res = train_ctc_one({all_train[i].clip_id: annotation for i in final},
                        all_train, dev,
                        cfg=CTCTrainConfig(seed=seed, max_epochs=max_epochs),
                        cache=cache, log_prefix=f"{log_prefix}[final] ")
    res.update(strategy="entropy_al", n_seed_clips=len(seed_idx),
               seed_budget_frac=seed_frac,
               seed_spent_hours=seed_frac * budget_s / 3600.0,
               selected_indices=final,
               total_minutes=(time.time() - t0) / 60.0)
    return res
