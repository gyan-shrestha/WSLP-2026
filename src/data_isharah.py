"""Isharah data layer: Saudi Sign Language, second corpus for the budget experiments.

Mirrors data_phoenix.py so every downstream component (selectors, featurizers, CTC
training, evaluation) works unchanged on either corpus. The only differences are where
clips and annotations come from.

Why this corpus matters here. All our budget results so far come from PHOENIX14T:
German Sign Language, weather forecasts, studio recording, nine signers. Weather
broadcasts are unusually repetitive, which plausibly favours typicality over
diversity, so a finding that typical clips beat rare ones is exactly the finding one
would expect to be corpus-specific. Isharah is a different sign language, recorded on
signers' own smartphones in unconstrained environments, with eighteen signers and a
broader domain. If the result holds on both, it is a property of the selection problem
rather than of weather reports.

Two official splits ship with the corpus and answer different questions:

    SI  signer-independent   dev and test signers never appear in training
    US  unseen sentences     sentences never appear in training, signers may

SI is the harder and more relevant setting for annotation planning, since a corpus
being built will be annotated for signers whose data already exists but generalised to
people it does not. SI is the default here for that reason.

Duration comes from the pose file's frame count. Isharah is recorded at 25 fps like
PHOENIX14T, so the annotator-hour cost model transfers without change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from data_phoenix import Clip, Vocab

FPS = 25.0


def load_annotations(ann_dir: Path, split_scheme: str = "SI") -> Dict[str, Dict[str, list]]:
    """Read `id|gloss|text` files into {split: {clip_id: {gloss, text}}}."""
    out: Dict[str, Dict[str, list]] = {}
    for split in ("train", "dev", "test"):
        p = ann_dir / f"{split_scheme}_{split}.txt"
        if not p.exists():
            continue
        rows: Dict[str, dict] = {}
        with open(p, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.rstrip("\n")
                if i == 0 and line.startswith("id|"):
                    continue
                parts = line.split("|")
                if len(parts) != 3:
                    continue
                cid, gloss, text = (x.strip() for x in parts)
                if not cid:
                    continue
                rows[cid] = {"gloss": gloss.split(), "text": text.split()}
        out[split] = rows
    return out


def load_split(pose_dir: Path, ann_dir: Path, split: str,
               split_scheme: str = "SI") -> List[Clip]:
    """Clips for one split, keeping only ids that have BOTH pose and annotation.

    The intersection matters: the pose file covers Isharah-2000 while the annotations
    we have are Isharah-1000, so a straight union would produce clips with no labels to
    reveal or labels with no video to select.
    """
    ann = load_annotations(ann_dir, split_scheme).get(split, {})
    clips: List[Clip] = []
    for cid, rec in sorted(ann.items()):
        p = pose_dir / f"{cid}.npz"
        if not p.exists():
            continue
        clips.append(Clip(
            clip_id=cid,
            signer=f"Signer{cid.split('_')[0]}",
            gloss=rec["gloss"],
            text=rec["text"],
            n_frames=0,
            pose_path=p,
        ))
    return clips


def fill_durations(clips: Sequence[Clip]) -> None:
    for c in clips:
        if c.n_frames == 0:
            z = np.load(c.pose_path, allow_pickle=True)
            try:
                c.n_frames = int(json.loads(str(z["meta"]))["n_frames"])
            except Exception:
                c.n_frames = int(z["pose"].shape[0])


def corpus_summary(pose_dir: Path, ann_dir: Path, split_scheme: str = "SI") -> dict:
    """Coverage report: how much of each split survives the pose/annotation join."""
    ann = load_annotations(ann_dir, split_scheme)
    out = {}
    for split, rows in ann.items():
        have = sum((pose_dir / f"{cid}.npz").exists() for cid in rows)
        clips = load_split(pose_dir, ann_dir, split, split_scheme)
        fill_durations(clips)
        hours = sum(c.n_frames for c in clips) / FPS / 3600.0
        out[split] = {
            "annotated": len(rows),
            "with_pose": have,
            "hours": round(hours, 2),
            "signers": len({c.signer for c in clips}),
            "gloss_types": len({t for c in clips for t in c.gloss}),
            "text_types": len({t for c in clips for t in c.text}),
        }
    return out
