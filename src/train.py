"""Train one SLT model on one purchased annotation set, and score it.

This is the inner loop of the budget experiment. A *purchase* is a set of
(clip, annotation_type) decisions produced by some acquisition strategy under an
annotator-hour budget. This module takes that purchase and answers the only question
that matters: what translation quality did those hours buy?

Two rules keep the comparison between strategies honest:

  1. Every clip in the purchase is trained on. A clip bought as `translation` gives
     text supervision; a clip bought as `gloss` gives text *and* CTC supervision. So
     a gloss purchase is strictly more informative per clip and strictly more
     expensive per clip, which is the tension the paper measures. Buying a gloss
     without the translation is not offered: in practice a glosser produces a gloss
     for a clip someone else can already translate, and modelling it otherwise would
     let gloss-only allocations "win" by dodging the text loss entirely.

  2. Everything not set by the purchase is held fixed, architecture, optimizer,
     schedule, decoding, seed. Strategies differ only in which clips were bought and
     at what fidelity.

Early stopping is on dev BLEU with a fixed patience. Budgets differ by orders of
magnitude, so a fixed epoch count would systematically underfit large budgets and
overfit small ones, which would manufacture exactly the curve shape we are trying
to measure.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_phoenix import (Clip, PhoenixDataset, Vocab, collate, FEATURE_DIM)
from models import ModelConfig, build


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 16
    max_epochs: int = 120
    patience: int = 12
    warmup: int = 300
    min_epochs: int = 10
    grad_clip: float = 1.0
    eval_every: int = 2
    seed: int = 0
    device: str = "cuda"
    amp: bool = True


def set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def corpus_bleu(hyps: Sequence[str], refs: Sequence[str]) -> Dict[str, float]:
    import sacrebleu
    b = sacrebleu.corpus_bleu(hyps, [refs])
    c = sacrebleu.corpus_chrf(hyps, [refs])
    return {"bleu": b.score, "chrf": c.score,
            "bleu1": b.precisions[0], "bleu4": b.precisions[3]}


@torch.no_grad()
def evaluate(model, loader, text_vocab: Vocab, device: str,
             max_len: int = 60) -> Tuple[Dict[str, float], List[str], List[str], List[str]]:
    model.eval()
    hyps, refs, signers = [], [], []
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model.generate(b, max_len=max_len)
        for i in range(out.size(0)):
            hyps.append(" ".join(text_vocab.decode(out[i].tolist())))
            refs.append(" ".join(text_vocab.decode(batch["text"][i].tolist())))
        signers.extend(batch["signer"])
    return corpus_bleu(hyps, refs), hyps, refs, signers


def per_signer_bleu(hyps, refs, signers) -> Dict[str, float]:
    """Fairness readout: quality broken down by signer. A selection strategy that
    concentrates its budget on a few signers can raise corpus BLEU while making the
    system worse for everyone else, and corpus-level BLEU hides that entirely."""
    out = {}
    for s in sorted(set(signers)):
        idx = [i for i, x in enumerate(signers) if x == s]
        if len(idx) >= 5:
            out[s] = corpus_bleu([hyps[i] for i in idx], [refs[i] for i in idx])["bleu"]
    return out


def train_one(
    purchase: Dict[str, str],          # clip_id -> "translation" | "gloss"
    all_train: Sequence[Clip],
    dev: Sequence[Clip],
    gloss_vocab: Vocab,
    text_vocab: Vocab,
    arch: str,                          # "gloss_supervised" | "gloss_free"
    tcfg: TrainConfig = TrainConfig(),
    mcfg: Optional[ModelConfig] = None,
    cache: Optional[dict] = None,
    log_prefix: str = "",
) -> Dict:
    set_seed(tcfg.seed)
    device = tcfg.device if torch.cuda.is_available() else "cpu"
    mcfg = mcfg or ModelConfig(feature_dim=FEATURE_DIM)

    bought = [c for c in all_train if c.clip_id in purchase]
    if not bought:
        raise ValueError("empty purchase")
    gloss_ids = {cid for cid, t in purchase.items() if t == "gloss"}

    tr_ds = PhoenixDataset(bought, gloss_vocab, text_vocab, cache=cache)
    dv_ds = PhoenixDataset(dev, gloss_vocab, text_vocab, cache=cache)
    tr = DataLoader(tr_ds, batch_size=tcfg.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=2, drop_last=False)
    dv = DataLoader(dv_ds, batch_size=32, shuffle=False, collate_fn=collate, num_workers=2)

    model = build(mcfg, len(text_vocab), len(gloss_vocab), arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    steps_per_epoch = max(1, len(tr))

    def lr_at(step):
        if step < tcfg.warmup:
            return step / max(tcfg.warmup, 1)
        total = tcfg.max_epochs * steps_per_epoch
        p = (step - tcfg.warmup) / max(total - tcfg.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=(tcfg.amp and device == "cuda"))

    best = {"bleu": -1.0}
    best_state, bad, step = None, 0, 0
    t0 = time.time()

    for ep in range(tcfg.max_epochs):
        model.train()
        tot = n = 0.0
        for batch in tr:
            b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
            gmask = torch.tensor([cid in gloss_ids for cid in batch["clip_id"]],
                                 dtype=torch.bool, device=device)
            with torch.amp.autocast("cuda", enabled=(tcfg.amp and device == "cuda")):
                out = model(b, gloss_supervised_mask=gmask)
                loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); n += 1; step += 1

        if (ep + 1) % tcfg.eval_every == 0 or ep == tcfg.max_epochs - 1:
            m, hyps, refs, sg = evaluate(model, dv, text_vocab, device)
            if m["bleu"] > best["bleu"]:
                best = dict(m, epoch=ep)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best["per_signer"] = per_signer_bleu(hyps, refs, sg)
                bad = 0
            else:
                bad += 1
            print(f"{log_prefix}ep{ep:3d} loss={tot/max(n,1):.3f} "
                  f"bleu={m['bleu']:.2f} chrf={m['chrf']:.2f} best={best['bleu']:.2f}",
                  flush=True)
            if bad >= tcfg.patience and ep >= tcfg.min_epochs:
                print(f"{log_prefix}early stop at ep{ep}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "arch": arch,
        "n_clips": len(bought),
        "n_gloss": len(gloss_ids),
        "n_translation": len(bought) - len(gloss_ids),
        "dev": {k: v for k, v in best.items() if k != "per_signer"},
        "per_signer_bleu": best.get("per_signer", {}),
        "minutes": (time.time() - t0) / 60.0,
    }
