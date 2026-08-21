"""Pose-to-gloss CTC recogniser, per the PoseCover spec (paper 5.5, impl. note 9.2).

Architecture, exactly as specified:
    frame projection to 256
    2 temporal convolution blocks (256 channels, kernel 5, stride 2, GELU,
        layer norm, dropout 0.1)
    4 Transformer encoder layers (256 hidden, 4 heads, ff 1024, dropout 0.1,
        pre-layer-norm, learned position embeddings)
    CTC output projection
    total under ~10M parameters

Why this exists alongside the translation model. The budget experiments so far
evaluate translation with BLEU, but the specified protocol evaluates gloss
recognition with WER. Those are not interchangeable: recognition depends far more on
discriminating individual signs, so a selection signal built from articulatory
structure could plausibly help there while failing on sentence-level translation. The
negative result on BLEU therefore does not transfer by assumption and has to be
re-measured on the specified task.

The two stride-2 convolutions downsample time by 4, which is what makes the CTC length
constraint T' >= L bite on short clips. Rows violating it are dropped from the loss
rather than silently producing inf, and the count is reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_phoenix import PAD


@dataclass
class CTCConfig:
    feature_dim: int = 139
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ff: int = 1024
    dropout: float = 0.1
    conv_kernel: int = 5
    conv_stride: int = 2
    n_conv_blocks: int = 2
    max_frames: int = 320          # impl. note 9.1
    max_len: int = 512


class TemporalConvBlock(nn.Module):
    def __init__(self, d: int, kernel: int, stride: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size=kernel, stride=stride,
                              padding=kernel // 2)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                      # (B, T, D)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.drop(self.norm(F.gelu(h)))


class PoseCTC(nn.Module):
    """Pose sequence -> gloss sequence, trained with CTC."""

    def __init__(self, cfg: CTCConfig, gloss_vocab: int):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.d_model), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.d_model))
        self.convs = nn.ModuleList([
            TemporalConvBlock(cfg.d_model, cfg.conv_kernel, cfg.conv_stride, cfg.dropout)
            for _ in range(cfg.n_conv_blocks)])
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)     # learned, per spec
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.ff, cfg.dropout,
            batch_first=True, norm_first=True)                 # pre-LN, per spec
        self.enc = nn.TransformerEncoder(layer, cfg.n_layers)
        self.out = nn.Linear(cfg.d_model, gloss_vocab)
        self.ctc = nn.CTCLoss(blank=PAD, zero_infinity=True)
        self.downsample = cfg.conv_stride ** cfg.n_conv_blocks

    def encode(self, x, x_mask):
        h = self.proj(x)
        for c in self.convs:
            h = c(h)
        T = h.size(1)
        h = h + self.pos(torch.arange(T, device=h.device))[None]
        # mask downsampled the same way the features were
        lens = x_mask.sum(1)
        new_lens = torch.clamp(torch.div(lens, self.downsample, rounding_mode="floor"), min=1)
        new_lens = torch.minimum(new_lens, torch.full_like(new_lens, T))
        m = torch.arange(T, device=h.device)[None] < new_lens[:, None]
        return self.enc(h, src_key_padding_mask=~m), new_lens

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        h, in_len = self.encode(batch["x"], batch["x_mask"])
        logp = F.log_softmax(self.out(h), dim=-1).transpose(0, 1)   # (T, B, V)

        g, gl = batch["gloss"], batch["gloss_len"].to(in_len.device)
        # CTC requires input length >= target length. Two stride-2 convolutions cut
        # time by 4, so short clips with long gloss sequences violate this. Drop those
        # rows rather than letting the loss go infinite, and report how many.
        keep = in_len >= gl
        out = {"n_dropped": int((~keep).sum())}
        if not keep.any():
            out["loss"] = logp.sum() * 0.0
            return out
        targets = torch.cat([g[i, :gl[i]] for i in range(g.size(0)) if keep[i]])
        loss = self.ctc(logp[:, keep], targets, in_len[keep], gl[keep])
        out["loss"] = loss if torch.isfinite(loss) else logp.sum() * 0.0
        return out

    @torch.no_grad()
    def decode_greedy(self, batch) -> List[List[int]]:
        """Greedy CTC decoding: argmax per frame, collapse repeats, drop blanks."""
        h, in_len = self.encode(batch["x"], batch["x_mask"])
        ids = self.out(h).argmax(-1)                                # (B, T)
        hyps = []
        for b in range(ids.size(0)):
            seq, prev = [], None
            for t in range(int(in_len[b])):
                k = int(ids[b, t])
                if k != prev and k != PAD:
                    seq.append(k)
                prev = k
            hyps.append(seq)
        return hyps


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def levenshtein(ref: Sequence, hyp: Sequence):
    """Edit distance with the substitution / deletion / insertion breakdown that the
    spec asks to report separately."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i; op[i][0] = "D"
    for j in range(m + 1):
        d[0][j] = j; op[0][j] = "I"
    op[0][0] = None
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j], op[i][j] = d[i - 1][j - 1], "C"
            else:
                sub, dele, ins = d[i - 1][j - 1] + 1, d[i - 1][j] + 1, d[i][j - 1] + 1
                best = min(sub, dele, ins)
                d[i][j] = best
                op[i][j] = "S" if best == sub else ("D" if best == dele else "I")
    i, j = n, m
    s = de = ins = 0
    while i > 0 or j > 0:
        o = op[i][j]
        if o == "C":
            i, j = i - 1, j - 1
        elif o == "S":
            s += 1; i, j = i - 1, j - 1
        elif o == "D":
            de += 1; i -= 1
        else:
            ins += 1; j -= 1
    return s, de, ins


def corpus_wer(refs: Sequence[Sequence], hyps: Sequence[Sequence]) -> Dict[str, float]:
    """WER = (S + D + I) / N over the whole corpus, per paper Eq. 28.

    Reference tokens outside the selected subset's vocabulary stay in the reference
    and count as errors (impl. note 9.3). That is what makes vocabulary coverage part
    of what a selection strategy is being judged on, rather than something the
    evaluation quietly forgives.
    """
    S = D = I = N = 0
    for r, h in zip(refs, hyps):
        s, d, i = levenshtein(r, h)
        S += s; D += d; I += i; N += len(r)
    N = max(N, 1)
    return {"wer": 100.0 * (S + D + I) / N, "sub": 100.0 * S / N,
            "del": 100.0 * D / N, "ins": 100.0 * I / N, "n_ref_tokens": N}
