"""The two SLT architectures the budget experiments compare.

  GlossSupervisedSLT : shared encoder + CTC head on gloss + text decoder
  GlossFreeSLT       : shared encoder + text decoder only

They are deliberately the *same* network except for the CTC head. That is the whole
point of the comparison: if the gloss-supervised model wins, the win must come from
the gloss supervision itself and not from extra capacity, a different encoder, or a
different training recipe. Papers comparing published gloss-based and gloss-free
systems cannot make that claim, because those systems differ in a dozen ways at once
,  which is exactly what the CVIU reproducibility study found when it re-implemented
five gloss-free methods in one codebase and watched the reported gains shrink.

A clip that was allocated a translation but no gloss simply contributes no CTC term.
That is how "buying gloss for this clip" is simulated: the same model, trained on the
same clips, with gloss supervision present only where it was purchased.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_phoenix import BOS, EOS, PAD


@dataclass
class ModelConfig:
    feature_dim: int = 139   # data_phoenix.FEATURE_DIM: 25*2 + 21*2 + 21*2 + 1 + 4
    d_model: int = 256
    n_heads: int = 4
    n_enc_layers: int = 3
    n_dec_layers: int = 3
    ff: int = 1024
    dropout: float = 0.15
    max_len: int = 512


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SignEncoder(nn.Module):
    """Pose sequence -> contextual representations. Shared by both architectures."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.d_model), nn.ReLU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.pos = PositionalEncoding(cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.ff, cfg.dropout,
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.n_enc_layers)

    def forward(self, x, x_mask):
        h = self.norm(self.pos(self.proj(x)))
        return self.enc(h, src_key_padding_mask=~x_mask)


class TextDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, cfg.d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerDecoderLayer(
            cfg.d_model, cfg.n_heads, cfg.ff, cfg.dropout,
            batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, cfg.n_dec_layers)
        self.out = nn.Linear(cfg.d_model, vocab)
        self.d_model = cfg.d_model

    def forward(self, memory, mem_mask, tgt_in):
        t = tgt_in.size(1)
        causal = torch.triu(torch.ones(t, t, device=tgt_in.device, dtype=torch.bool), 1)
        h = self.drop(self.pos(self.emb(tgt_in) * math.sqrt(self.d_model)))
        h = self.dec(h, memory, tgt_mask=causal,
                     tgt_key_padding_mask=(tgt_in == PAD),
                     memory_key_padding_mask=~mem_mask)
        return self.out(h)


class SLTModel(nn.Module):
    """use_gloss=False gives the gloss-free architecture; everything else is identical."""

    def __init__(self, cfg: ModelConfig, text_vocab: int,
                 gloss_vocab: int = 0, use_gloss: bool = True,
                 ctc_weight: float = 0.4, label_smoothing: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.use_gloss = use_gloss and gloss_vocab > 0
        self.ctc_weight = ctc_weight
        self.encoder = SignEncoder(cfg)
        self.decoder = TextDecoder(cfg, text_vocab)
        self.ctc_head = nn.Linear(cfg.d_model, gloss_vocab) if self.use_gloss else None
        self.ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=label_smoothing)
        self.ctc = nn.CTCLoss(blank=PAD, zero_infinity=True) if self.use_gloss else None

    def forward(self, batch, gloss_supervised_mask=None):
        """gloss_supervised_mask: (B,) bool. True where a gloss was actually purchased
        for this clip. Clips without a purchased gloss contribute translation loss
        only, this is the mechanism by which an allocation is realised in training."""
        x, x_mask = batch["x"], batch["x_mask"]
        mem = self.encoder(x, x_mask)

        tgt = batch["text"]
        logits = self.decoder(mem, x_mask, tgt[:, :-1])
        loss_txt = self.ce(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

        out = {"loss_text": loss_txt, "loss": loss_txt}
        if not self.use_gloss:
            return out

        if gloss_supervised_mask is None:
            gloss_supervised_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        if gloss_supervised_mask.any():
            sel = gloss_supervised_mask
            logp = F.log_softmax(self.ctc_head(mem[sel]), dim=-1).transpose(0, 1)
            in_len = x_mask[sel].sum(1).to(torch.long)
            g, gl = batch["gloss"][sel], batch["gloss_len"][sel]
            targets = torch.cat([g[i, :gl[i]] for i in range(g.size(0))]) if g.size(0) else g.new_zeros(0)
            # CTC needs input length >= target length; drop the pathological rows
            keep = in_len >= gl.to(in_len.device)
            if keep.any() and targets.numel():
                loss_ctc = self.ctc(logp[:, keep], targets, in_len[keep], gl.to(in_len.device)[keep])
                if torch.isfinite(loss_ctc):
                    out["loss_ctc"] = loss_ctc
                    out["loss"] = loss_txt + self.ctc_weight * loss_ctc
        return out

    @torch.no_grad()
    def generate(self, batch, max_len: int = 60):
        """Greedy decoding. Beam search is a quality knob, not a fairness knob, every
        strategy is decoded identically, so greedy keeps the budget grid affordable."""
        x, x_mask = batch["x"], batch["x_mask"]
        mem = self.encoder(x, x_mask)
        B = x.size(0)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=x.device)
        done = torch.zeros(B, dtype=torch.bool, device=x.device)
        for _ in range(max_len):
            nxt = self.decoder(mem, x_mask, ys)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
            done |= nxt == EOS
            if done.all():
                break
        return ys[:, 1:]


def build(cfg: ModelConfig, text_vocab: int, gloss_vocab: int, arch: str) -> SLTModel:
    assert arch in ("gloss_supervised", "gloss_free")
    return SLTModel(cfg, text_vocab, gloss_vocab, use_gloss=(arch == "gloss_supervised"))
