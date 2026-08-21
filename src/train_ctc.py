"""Train the pose-to-gloss CTC recogniser on one purchased subset and score it by WER.

Implements the vocabulary policy from impl. note 9.3, which the earlier translation
experiments did not follow and which is the change most likely to move the numbers:

    the output vocabulary is built ONLY from glosses present in the selected subset

Previously the vocabulary came from the whole training pool, so every model could
predict words it had never paid to annotate. That is a quiet form of label leakage: it
hands each strategy the same free knowledge of the label space and hides exactly the
thing a selection method should be judged on, namely whether it bought the vocabulary
it needs. Under the correct policy, glosses missing from the purchase remain in the
reference and count as deletions, so vocabulary coverage shows up directly in WER.

One consequence worth expecting: absolute WER will be much worse than numbers reported
against a full-pool vocabulary, especially at small budgets. That is the point, not a
regression.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_phoenix import Clip, PhoenixDataset, Vocab, collate, FEATURE_DIM
from pose_ctc import CTCConfig, PoseCTC, corpus_wer


@dataclass
class CTCTrainConfig:
    lr: float = 3e-4                # impl. note 10
    weight_decay: float = 1e-2
    batch_size: int = 8
    grad_accum: int = 4             # effective batch 32
    max_epochs: int = 60
    patience: int = 8               # evaluations
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda"
    amp: bool = True


def build_subset_vocab(bought: Sequence[Clip]) -> Vocab:
    """Impl. note 9.3: vocabulary from the selected subset only."""
    return Vocab([t for c in bought for t in c.gloss])


def oov_rate(dev: Sequence[Clip], vocab: Vocab) -> float:
    """Fraction of dev gloss tokens absent from the purchased vocabulary. These are
    unreachable by construction and become deletions in WER."""
    tot = miss = 0
    for c in dev:
        for t in c.gloss:
            tot += 1
            miss += t not in vocab.stoi
    return 100.0 * miss / max(tot, 1)


def set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, vocab: Vocab, dev_clips: Sequence[Clip], device: str):
    """Decode, then score against the ORIGINAL gloss strings.

    References are the true glosses, not the vocabulary-mapped ids, so a gloss the
    purchase never bought cannot be silently dropped from the reference.
    """
    model.eval()
    hyps: List[List[str]] = []
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        for seq in model.decode_greedy(b):
            hyps.append([vocab.itos[i] for i in seq if i < len(vocab)])
    refs = [c.gloss for c in dev_clips]
    return corpus_wer(refs, hyps), hyps


def train_ctc_one(
    purchase: Dict[str, str],
    all_train: Sequence[Clip],
    dev: Sequence[Clip],
    cfg: CTCTrainConfig = CTCTrainConfig(),
    cache: Optional[dict] = None,
    log_prefix: str = "",
    return_model: bool = False,
) -> Dict:
    """With return_model=True the result carries the trained model and its vocabulary
    under "_model" and "_vocab". Active learning needs the seed model itself in order
    to score the pool, and reconstructing one from the returned metrics would give an
    untrained network whose entropies are meaningless."""
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    bought = [c for c in all_train if c.clip_id in purchase]
    if not bought:
        raise ValueError("empty purchase")

    gv = build_subset_vocab(bought)          # <-- the policy
    oov = oov_rate(dev, gv)

    dummy_text = Vocab(["x"])
    tr_ds = PhoenixDataset(bought, gv, dummy_text, max_frames=320, cache=cache)
    dv_ds = PhoenixDataset(dev, gv, dummy_text, max_frames=320, cache=cache)
    tr = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=2)
    dv = DataLoader(dv_ds, batch_size=16, shuffle=False, collate_fn=collate, num_workers=2)

    model = PoseCTC(CTCConfig(feature_dim=FEATURE_DIM), len(gv)).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = max(1, len(tr) // cfg.grad_accum) * cfg.max_epochs
    warm = max(1, int(cfg.warmup_frac * steps))

    def lr_at(s):
        if s < warm:
            return s / warm
        return 0.5 * (1 + math.cos(math.pi * min((s - warm) / max(steps - warm, 1), 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device == "cuda"))

    best = {"wer": 1e9}
    best_state, bad, step, dropped = None, 0, 0, 0
    t0 = time.time()

    for ep in range(cfg.max_epochs):
        model.train()
        tot = n = 0.0
        opt.zero_grad(set_to_none=True)
        for k, batch in enumerate(tr):
            b = {kk: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for kk, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(cfg.amp and device == "cuda")):
                out = model(b)
                loss = out["loss"] / cfg.grad_accum
            scaler.scale(loss).backward()
            dropped += out["n_dropped"]
            if (k + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True); step += 1
            tot += float(out["loss"]); n += 1

        m, _ = evaluate(model, dv, gv, dev, device)
        if m["wer"] < best["wer"]:
            best = dict(m, epoch=ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"{log_prefix}ep{ep:3d} loss={tot/max(n,1):.3f} "
              f"wer={m['wer']:.2f} best={best['wer']:.2f}", flush=True)
        if bad >= cfg.patience:
            print(f"{log_prefix}early stop at ep{ep}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    out = {
        "task": "gloss_ctc",
        "n_clips": len(bought),
        "gloss_vocab_size": len(gv),
        "dev_oov_token_pct": oov,
        "ctc_rows_dropped": int(dropped),
        "dev": best,
        "params_M": n_par,
        "minutes": (time.time() - t0) / 60.0,
    }
    if return_model:
        # underscore-prefixed so json.dumps of the metrics never tries to serialise them
        out["_model"] = model
        out["_vocab"] = gv
    return out
