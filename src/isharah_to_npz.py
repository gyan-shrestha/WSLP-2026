"""Convert the Isharah pose PKL into the same per-clip npz layout that
extract_pose.py emits for PHOENIX14T, so every downstream consumer
(PhonologicalFeaturizer, cluster_gate.py, the allocator) works on both corpora
without a corpus-specific branch.

Isharah PKL layout (per the official reader notebook):
    pose_dict[sample_id]['keypoints'] -> (T, J, 2)
      0:21   right hand      (MediaPipe hand topology)
      21:42  left hand
      42:61  lips            (19 landmarks, not a full face mesh)
      61:    body            (25 joints = MediaPipe pose landmarks 0..24)

The body slice is the happy accident that makes this cheap: UPPER_BODY_IDXS in the
official notebook is the contiguous range 0..24, so body-local index i *is*
MediaPipe pose index i. PoseLayout's defaults (nose 0, shoulders 11/12, wrists
15/16) therefore apply to the body slice unchanged, no index remapping needed.

Two differences from the PHOENIX14T path that callers must not ignore:

  1. 2D only. There is no z channel. The featurizer already slices [:, :, :2]
     everywhere, so this is harmless, but the npz carries z=0 to keep shapes uniform.

  2. No detection mask. Isharah ships keypoints without per-frame confidence, so
     "was this hand detected" has to be inferred. Missing hands are conventionally
     encoded as all-zero (or all-equal) coordinates, so a frame is marked
     undetected when the hand's coordinates have effectively zero spread. This
     matters: PHOENIX14T's real detection rate is ~0.69, and if Isharah frames were
     silently treated as always-detected the two corpora would not be comparable.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

N_RH, N_LH, N_LIPS = 21, 21, 19
RH = slice(0, N_RH)
LH = slice(N_RH, N_RH + N_LH)
LIPS = slice(N_RH + N_LH, N_RH + N_LH + N_LIPS)
BODY = slice(N_RH + N_LH + N_LIPS, None)


def _to3(a: np.ndarray) -> np.ndarray:
    """(T, J, 2) -> (T, J, 3) with z=0, matching the PHOENIX14T npz shape."""
    return np.concatenate([a.astype(np.float32),
                           np.zeros((*a.shape[:2], 1), np.float32)], axis=-1)


def _detected(part: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-frame detection flag for a (T, J, 2) part.

    Absent keypoints are written as constant (usually zero) coordinates, which
    collapses the per-frame spread to ~0. Real articulated hands always have
    spread. Threshold on coordinate range rather than on exact zeros, since some
    exporters emit small constants or NaN instead.
    """
    if part.size == 0:
        return np.zeros(part.shape[0], bool)
    p = np.nan_to_num(part, nan=0.0)
    spread = p.reshape(p.shape[0], -1).max(1) - p.reshape(p.shape[0], -1).min(1)
    return spread > eps


def convert(pkl_path: Path, out_dir: Path, limit: int = 0, ann: dict | None = None):
    print(f"loading {pkl_path} ({pkl_path.stat().st_size / 1e9:.1f} GB) ...", flush=True)
    with open(pkl_path, "rb") as fh:
        pose_dict = pickle.load(fh)
    print(f"  {len(pose_dict)} samples", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(pose_dict.keys())
    if limit:
        ids = ids[:limit]

    n_body_seen, det_acc, written = set(), [], 0
    for k, sid in enumerate(ids):
        kp = np.asarray(pose_dict[sid]["keypoints"])
        if kp.ndim != 3 or kp.shape[-1] != 2:
            print(f"  SKIP {sid}: unexpected shape {kp.shape}")
            continue
        body = kp[:, BODY, :]
        n_body_seen.add(body.shape[1])

        rh_raw, lh_raw, lips_raw = kp[:, RH, :], kp[:, LH, :], kp[:, LIPS, :]
        ok = np.stack([_detected(body), _detected(lh_raw),
                       _detected(rh_raw), _detected(lips_raw)], axis=1)
        det_acc.append(ok.mean(0))

        meta = {"clip_id": str(sid), "signer": _signer_of(sid, ann),
                "split": "unknown", "n_frames": int(kp.shape[0]),
                "gloss": (ann or {}).get(str(sid), {}).get("gloss", ""),
                "translation": (ann or {}).get(str(sid), {}).get("translation", "")}
        np.savez_compressed(out_dir / f"{sid}.npz", meta=json.dumps(meta),
                            pose=_to3(body), lh=_to3(lh_raw), rh=_to3(rh_raw),
                            face=_to3(lips_raw), ok=ok)
        written += 1
        if written % 2000 == 0:
            print(f"  [{k+1}/{len(ids)}] written={written}", flush=True)

    D = np.array(det_acc)
    print(f"\nwrote {written} clips to {out_dir}")
    print(f"body joint counts seen: {sorted(n_body_seen)}  (expect {{25}})")
    if 25 not in n_body_seen:
        print("  WARNING: body is not the expected 25-joint MediaPipe upper-body "
              "subset. PoseLayout indices (shoulders 11/12, wrists 15/16) may be wrong.")
    print("mean per-frame detection:")
    for i, name in enumerate(["body", "left_hand", "right_hand", "lips"]):
        print(f"  {name:11s} {D[:, i].mean():.3f}")
    return written


def _signer_of(sid, ann):
    if ann and str(sid) in ann and "signer" in ann[str(sid)]:
        return ann[str(sid)]["signer"]
    # Isharah sample ids look like '00_0001'; the leading field is the signer.
    s = str(sid)
    return f"Signer{s.split('_')[0]}" if "_" in s else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--annotations", default="",
                    help="optional json {sample_id: {gloss, translation, signer}}")
    args = ap.parse_args()

    ann = None
    if args.annotations:
        with open(args.annotations) as fh:
            ann = json.load(fh)
        print(f"loaded annotations for {len(ann)} samples")

    convert(Path(args.pkl), Path(args.out), args.limit, ann)
    return 0


if __name__ == "__main__":
    sys.exit(main())
