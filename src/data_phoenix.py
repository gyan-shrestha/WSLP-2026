"""PHOENIX14T data layer for the budget-allocation experiments.

Feeds two architectures from one representation:
  - gloss-supervised : pose -> gloss (CTC) -> text
  - gloss-free       : pose -> text directly

Input is the pose we already extracted, not raw frames. That choice is deliberate
and has to be defended in the paper: pose-based SLT scores below video-based SLT in
absolute BLEU, but the experiments here compare *allocation strategies under a fixed
annotator-hour budget*, and every strategy is evaluated with the same encoder. What
matters is where the curves cross, not their height. Pose also makes the budget grid
affordable, dozens of runs across budgets, strategies and seeds instead of a handful.

Detection matters. Hand landmarks are missing in ~23% of frames (and the miss rate
varies by signer, see the fairness section), so frames carry their detection mask
into the model rather than being silently treated as zeros, otherwise "hand at the
origin" and "hand not found" become the same input.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# MediaPipe pose indices, valid on both PHOENIX14T (33 joints) and Isharah (25).
UPPER_BODY = list(range(25))
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16


class Vocab:
    def __init__(self, tokens: Sequence[str], min_freq: int = 1):
        from collections import Counter
        c = Counter(tokens)
        self.itos = list(SPECIALS) + sorted(w for w, n in c.items() if n >= min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks: Sequence[str], bos=False, eos=False) -> List[int]:
        ids = [self.stoi.get(t, UNK) for t in toks]
        if bos:
            ids = [BOS] + ids
        if eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.itos[i] for i in ids
                if i not in (PAD, BOS, EOS) and i < len(self.itos)]


@dataclass
class Clip:
    clip_id: str
    signer: str
    gloss: List[str]
    text: List[str]
    n_frames: int
    pose_path: Path

    @property
    def duration_s(self) -> float:
        """PHOENIX14T is 25 fps. Duration drives annotation cost, so this is the
        quantity the budget is denominated in, not clip count."""
        return self.n_frames / 25.0


def load_split(root: Path, poses: Path, split: str) -> List[Clip]:
    csv_path = root / "annotations" / "manual" / f"PHOENIX-2014-T.{split}.corpus.csv"
    out: List[Clip] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="|"):
            cid = r["name"].strip()
            p = poses / split / f"{cid}.npz"
            if not p.exists():
                continue
            out.append(Clip(
                clip_id=cid,
                signer=r["speaker"].strip(),
                gloss=r["orth"].strip().split(),
                text=r["translation"].strip().lower().split(),
                n_frames=0,          # filled lazily on first load
                pose_path=p,
            ))
    return out


def fill_durations(clips: Sequence[Clip]) -> None:
    """Read n_frames from each npz meta. Needed before any budgeting, since cost is
    proportional to video duration."""
    for c in clips:
        if c.n_frames == 0:
            z = np.load(c.pose_path, allow_pickle=True)
            c.n_frames = int(json.loads(str(z["meta"]))["n_frames"])


# ---------------------------------------------------------------------------
# pose -> model features
# ---------------------------------------------------------------------------

def pose_features(npz_path: Path) -> np.ndarray:
    """(T, D) float32 features from one clip's npz.

    Normalization is shoulder-centred and shoulder-width-scaled, which removes
    camera distance and signer position, the same normalization the articulatory
    featurizer uses, so the model and the coverage measure see a consistent space.

    Layout: body(25x2) + left hand(21x2) + right hand(21x2) + lips-or-face energy(1)
            + detection flags(4)  =  135 dims
    """
    z = np.load(npz_path, allow_pickle=True)
    pose, lh, rh, face, ok = z["pose"], z["lh"], z["rh"], z["face"], z["ok"]

    body = pose[:, UPPER_BODY, :2].astype(np.float32)
    mid = (body[:, L_SHOULDER] + body[:, R_SHOULDER]) / 2.0
    width = np.linalg.norm(body[:, L_SHOULDER] - body[:, R_SHOULDER], axis=-1)
    width = np.maximum(width, 1e-3)[:, None]

    def norm(x):
        return (x[:, :, :2].astype(np.float32) - mid[:, None, :]) / width[:, None, :]

    feats = [norm(body).reshape(len(body), -1),
             norm(lh).reshape(len(lh), -1),
             norm(rh).reshape(len(rh), -1)]

    # non-manual activity: frame-to-frame facial displacement energy, one scalar
    f = face[:, :, :2].astype(np.float32)
    e = np.zeros((len(f), 1), np.float32)
    if len(f) > 1:
        e[1:, 0] = np.linalg.norm(np.diff(f, axis=0), axis=-1).mean(-1)
    feats.append(e)

    # detection flags are input, not silently-zeroed coordinates
    feats.append(ok.astype(np.float32))
    return np.nan_to_num(np.concatenate(feats, axis=1), nan=0.0)


FEATURE_DIM = 25 * 2 + 21 * 2 + 21 * 2 + 1 + 4


class PhoenixDataset(Dataset):
    def __init__(self, clips: Sequence[Clip], gloss_vocab: Vocab, text_vocab: Vocab,
                 max_frames: int = 400, cache: Optional[Dict[str, np.ndarray]] = None):
        self.clips = list(clips)
        self.gv, self.tv = gloss_vocab, text_vocab
        self.max_frames = max_frames
        self.cache = cache if cache is not None else {}

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        c = self.clips[i]
        if c.clip_id not in self.cache:
            self.cache[c.clip_id] = pose_features(c.pose_path)
        x = self.cache[c.clip_id]
        if len(x) > self.max_frames:                       # uniform subsample, keeps span
            idx = np.linspace(0, len(x) - 1, self.max_frames).astype(int)
            x = x[idx]
        return {
            "x": torch.from_numpy(x),
            "gloss": torch.tensor(self.gv.encode(c.gloss), dtype=torch.long),
            "text": torch.tensor(self.tv.encode(c.text, bos=True, eos=True), dtype=torch.long),
            "clip_id": c.clip_id,
            "signer": c.signer,
        }


def collate(batch):
    B = len(batch)
    T = max(b["x"].shape[0] for b in batch)
    D = batch[0]["x"].shape[1]
    x = torch.zeros(B, T, D)
    x_mask = torch.zeros(B, T, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["x"].shape[0]
        x[i, :t] = b["x"]
        x_mask[i, :t] = True

    def pad(key):
        L = max(len(b[key]) for b in batch)
        out = torch.full((B, L), PAD, dtype=torch.long)
        lens = torch.zeros(B, dtype=torch.long)
        for i, b in enumerate(batch):
            out[i, :len(b[key])] = b[key]
            lens[i] = len(b[key])
        return out, lens

    gloss, gloss_len = pad("gloss")
    text, text_len = pad("text")
    return {"x": x, "x_mask": x_mask, "gloss": gloss, "gloss_len": gloss_len,
            "text": text, "text_len": text_len,
            "clip_id": [b["clip_id"] for b in batch],
            "signer": [b["signer"] for b in batch]}


def build_vocabs(train: Sequence[Clip], min_freq: int = 1):
    gv = Vocab([t for c in train for t in c.gloss], min_freq)
    tv = Vocab([t for c in train for t in c.text], min_freq)
    return gv, tv
