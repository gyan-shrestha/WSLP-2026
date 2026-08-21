"""Pose extraction for PHOENIX14T -> per-clip npz consumable by PhonologicalFeaturizer.

PHOENIX14T v3 ships pre-extracted frames, so there is no video decoding:
    features/fullFrame-210x260px/<split>/<clip_id>/*.png
    annotations/manual/PHOENIX-2014-T.<split>.corpus.csv   (name|video|start|end|speaker|orth|translation)

Output, one npz per clip:
    pose  (T, 33, 3)   MediaPipe Holistic body landmarks, normalized image coords + z
    lh    (T, 21, 3)   left hand   (zeros + lh_ok=False where not detected)
    rh    (T, 21, 3)   right hand
    face  (T, F, 3)    face mesh
    ok    (T, 4) bool  per-frame detection flags [pose, lh, rh, face]
    meta            clip_id, split, signer, n_frames, gloss, translation

`signer` is carried through deliberately: the week-2 gate is whether discovered
clusters track handshape/movement or merely signer identity and camera setup, and
that check is impossible without signer labels attached to every clip.

MediaPipe >= 1.0 removed mp.solutions.holistic; this uses the Tasks API
(vision.HolisticLandmarker), which yields the same 33/21/21 layout PoseLayout assumes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

N_POSE, N_HAND = 33, 21


def load_corpus(ann_dir: Path, split: str):
    """Return list of dicts from the PHOENIX14T corpus csv (pipe-separated)."""
    f = ann_dir / f"PHOENIX-2014-T.{split}.corpus.csv"
    rows = []
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="|"):
            rows.append({
                "clip_id": r["name"].strip(),
                "signer": r["speaker"].strip(),
                "gloss": r["orth"].strip(),
                "translation": r["translation"].strip(),
            })
    return rows


def build_landmarker(model_path: Path):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(opts)


def _to_array(landmarks, n_expected):
    """MediaPipe landmark list -> (n, 3); zeros when the part was not detected."""
    if not landmarks:
        return np.zeros((n_expected, 3), np.float32), False
    a = np.array([[p.x, p.y, p.z] for p in landmarks], np.float32)
    if a.shape[0] != n_expected and n_expected > 0:
        # face mesh count varies by model revision; keep whatever the model gives
        pass
    return a, True


def extract_clip(landmarker, frame_paths, ts0=0, fps=25, upscale=1.0):
    """ts0: global timestamp offset in ms. A single landmarker is reused across
    clips, and MediaPipe's VIDEO mode requires timestamps to increase monotonically
    over the whole session, not per clip, so the caller threads a running offset
    through rather than restarting at zero each time.

    upscale: PHOENIX14T frames are 210x260, which puts a signer's hand at roughly
    30x30 px, below what the hand landmarker reliably detects (measured: 0.69
    detection rate on the dominant hand at native resolution). Upscaling before
    landmarking recovers detections. Landmarks come back in normalized [0,1] image
    coordinates, so the output is scale-invariant and needs no rescaling back.
    """
    import cv2
    import mediapipe as mp

    pose_s, lh_s, rh_s, face_s, ok_s = [], [], [], [], []
    n_face = None
    for i, fp in enumerate(frame_paths):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if upscale != 1.0:
            img = cv2.resize(img, None, fx=upscale, fy=upscale,
                             interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_img, ts0 + int(i * 1000 / fps))

        p, p_ok = _to_array(getattr(res, "pose_landmarks", None), N_POSE)
        l, l_ok = _to_array(getattr(res, "left_hand_landmarks", None), N_HAND)
        r, r_ok = _to_array(getattr(res, "right_hand_landmarks", None), N_HAND)
        f_raw = getattr(res, "face_landmarks", None)
        f, f_ok = _to_array(f_raw, 0 if n_face is None else n_face)
        if f_ok and n_face is None:
            n_face = f.shape[0]
        if not f_ok:
            f = np.zeros((n_face or 1, 3), np.float32)

        pose_s.append(p if p_ok else np.zeros((N_POSE, 3), np.float32))
        lh_s.append(l); rh_s.append(r); face_s.append(f)
        ok_s.append([p_ok, l_ok, r_ok, f_ok])

    if not pose_s:
        return None
    # pad face rows to a common count (model can drop the mesh on some frames)
    fmax = max(x.shape[0] for x in face_s)
    face_s = [np.pad(x, ((0, fmax - x.shape[0]), (0, 0))) for x in face_s]
    return {
        "pose": np.stack(pose_s), "lh": np.stack(lh_s),
        "rh": np.stack(rh_s), "face": np.stack(face_s),
        "ok": np.array(ok_s, bool),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="PHOENIX14T root (contains features/ and annotations/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="holistic_landmarker.task")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N clips")
    ap.add_argument("--upscale", type=float, default=1.0,
                    help="resize factor applied before landmarking (see extract_clip)")
    args = ap.parse_args()

    root = Path(args.root)
    frames_root = root / "features" / "fullFrame-210x260px" / args.split
    ann_dir = root / "annotations" / "manual"
    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_corpus(ann_dir, args.split)
    rows = rows[args.shard::args.n_shards]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[shard {args.shard}/{args.n_shards}] {len(rows)} clips from {args.split}", flush=True)

    landmarker = build_landmarker(Path(args.model))
    done = skipped = 0
    ts = 0   # global monotonic timestamp (ms) across all clips in this shard
    for k, row in enumerate(rows):
        dst = out_dir / f"{row['clip_id']}.npz"
        if dst.exists():
            skipped += 1
            continue
        clip_dir = frames_root / row["clip_id"]
        frames = sorted(clip_dir.glob("*.png")) or sorted(clip_dir.glob("*.jpg"))
        if not frames:
            print(f"  MISSING FRAMES {row['clip_id']}", flush=True)
            continue
        out = extract_clip(landmarker, frames, ts0=ts, upscale=args.upscale)
        ts += int(len(frames) * 1000 / 25) + 1000   # advance past this clip, plus a gap
        if out is None:
            print(f"  EMPTY {row['clip_id']}", flush=True)
            continue
        meta = dict(row, split=args.split, n_frames=int(out["pose"].shape[0]))
        np.savez_compressed(dst, meta=json.dumps(meta), **out)
        done += 1
        if done % 25 == 0:
            det = out["ok"].mean(0)
            print(f"  [{k+1}/{len(rows)}] {row['clip_id']} T={meta['n_frames']} "
                  f"det(pose/lh/rh/face)={np.round(det,2).tolist()}", flush=True)

    print(f"[shard {args.shard}] done={done} skipped={skipped}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
