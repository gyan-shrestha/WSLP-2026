"""Articulatory descriptors exactly as specified (paper Eq. 4, 7; impl. note 4.2, 6, 6.1, 6.2).

Six groups, replacing the five-group approximation in descriptors.py:

    H  hand configuration   64 codewords
    O  orientation          16
    L  location             24
    M  motion               32
    C  coordination          8
    N  non-manual           16

Three things the earlier version left out, each of which changes what the codebooks
learn:

  Shoulder rotation (Eq. 4). Translating and scaling removes camera distance, but not
  body roll. Without rotating the shoulder axis to horizontal, a signer who leans
  produces different location codewords for the same articulation.

  Stable-frame rule (6.1). Handshape, orientation and location are properties of held
  configurations, not of the transitions between them. Quantising every frame lets
  motion blur and mid-transition hand shapes populate the codebook with configurations
  no signer ever holds. Frames below the clip's 60th speed percentile are kept for
  those three groups; motion and coordination use all valid frames, since transitions
  are exactly what they measure.

  Hand canonicalisation (6). Each hand is translated to its wrist, scaled by palm size,
  rotated so the wrist-to-middle-MCP axis is common, and left hands are mirrored into
  the right-hand frame. Without this, the same handshape on the left and right hand
  lands in different codewords and the inventory doubles for no linguistic reason.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# MediaPipe hand landmarks
WRIST = 0
MCP = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
TIPS = [4, 8, 12, 16, 20]
# MediaPipe pose landmarks
NOSE, L_SH, R_SH, L_WR, R_WR = 0, 11, 12, 15, 16
L_EYE, R_EYE, MOUTH_L, MOUTH_R = 2, 5, 9, 10

GROUPS = ("H", "O", "L", "M", "C", "N")
N_CODEWORDS = {"H": 64, "O": 16, "L": 24, "M": 32, "C": 8, "N": 16}   # impl. note 6


@dataclass
class PCDescriptorConfig:
    n_codewords: Dict[str, int] = field(default_factory=lambda: dict(N_CODEWORDS))
    stable_percentile: float = 60.0      # 6.1
    motion_window: int = 8               # 6
    coord_window: int = 16               # 6
    median_filter: int = 5               # 4.2
    max_gap_interp: int = 5              # 4.2
    clip_range: float = 5.0              # 4.2
    hand_pca_dim: int = 20               # 6
    max_desc_per_clip: int = 200         # 6.2
    max_desc_pool: int = 200_000         # 6.2
    kmeans_batch: int = 4096             # 6.2
    min_face_frac: float = 0.70          # 6, disable N below this
    seed: int = 0


# ---------------------------------------------------------------------------
# 4.2 normalization
# ---------------------------------------------------------------------------

def _median_filter(x: np.ndarray, w: int) -> np.ndarray:
    if w < 2 or len(x) < w:
        return x
    pad = w // 2
    p = np.pad(x, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    return np.stack([np.median(p[i:i + w], axis=0) for i in range(len(x))])


def _interp_gaps(x: np.ndarray, valid: np.ndarray, max_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interpolation across gaps of at most max_gap frames. Longer gaps stay
    masked, so a hand missing for a second is never invented."""
    x, valid = x.copy(), valid.copy()
    T = len(x)
    i = 0
    while i < T:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < T and not valid[j]:
            j += 1
        if 0 < i and j < T and (j - i) <= max_gap:
            a, b = x[i - 1], x[j]
            for k in range(i, j):
                t = (k - i + 1) / (j - i + 1)
                x[k] = (1 - t) * a + t * b
            valid[i:j] = True
        i = j
    return x, valid


def normalize_pose(pose: np.ndarray, cfg: PCDescriptorConfig):
    """Eq. 3-4 plus impl. note 4.2: centre, scale, ROTATE, clip, filter.

    Returns normalized (T, J, 2), the rotation matrices, shoulder mid and width.
    """
    p = pose[:, :, :2].astype(np.float32)
    mid = (p[:, L_SH] + p[:, R_SH]) / 2.0
    axis = p[:, L_SH] - p[:, R_SH]
    width = np.maximum(np.linalg.norm(axis, axis=-1), 1e-3)

    # rotate the shoulder axis to horizontal (Eq. 4); this is what the earlier
    # implementation omitted
    ang = np.arctan2(axis[:, 1], axis[:, 0])
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)   # (T,2,2)

    out = np.einsum("tij,tkj->tki", R, p - mid[:, None, :]) / width[:, None, None]
    out = np.clip(out, -cfg.clip_range, cfg.clip_range)
    out = _median_filter(out, cfg.median_filter)
    return out.astype(np.float32), R, mid, width


def normalize_hand_to_body(hand: np.ndarray, R, mid, width, cfg):
    h = hand[:, :, :2].astype(np.float32)
    out = np.einsum("tij,tkj->tki", R, h - mid[:, None, :]) / width[:, None, None]
    return np.clip(out, -cfg.clip_range, cfg.clip_range).astype(np.float32)


# ---------------------------------------------------------------------------
# 6.1 stable-frame rule
# ---------------------------------------------------------------------------

def stable_frames(wrist_xy: np.ndarray, centroid_xy: np.ndarray,
                  pct: float, smooth: int = 5) -> np.ndarray:
    """Frames below the clip's `pct` speed percentile, i.e. held configurations."""
    T = len(wrist_xy)
    if T < 3:
        return np.ones(T, bool)
    sp = (np.linalg.norm(np.diff(wrist_xy, axis=0, prepend=wrist_xy[:1]), axis=-1)
          + np.linalg.norm(np.diff(centroid_xy, axis=0, prepend=centroid_xy[:1]), axis=-1))
    k = max(1, smooth)
    sm = np.convolve(sp, np.ones(k) / k, mode="same")
    return sm <= np.percentile(sm, pct)


# ---------------------------------------------------------------------------
# per-group descriptors
# ---------------------------------------------------------------------------

def hand_configuration(hand_xy: np.ndarray, is_left: bool) -> np.ndarray:
    """Canonicalised handshape: wrist-centred, palm-scaled, rotation-aligned, and
    left hands mirrored into the right-hand coordinate system (impl. note 6).

    Accepts (T, 21, 2) or (T, 21, 3); the stored pose carries a z channel that
    MediaPipe populates but which is not metrically comparable across clips, so only
    x and y are used.
    """
    h = hand_xy[:, :, :2].astype(np.float32)
    h = h - h[:, WRIST:WRIST + 1, :]
    if is_left:
        h = h * np.array([-1.0, 1.0], np.float32)     # mirror into right-hand frame
    palm = np.maximum(np.linalg.norm(h[:, MCP["middle"]], axis=-1, keepdims=True), 1e-4)
    h = h / palm[:, :, None] if palm.ndim == 2 else h / palm[:, None]
    ax = h[:, MCP["middle"]]
    ang = np.arctan2(ax[:, 1], ax[:, 0]) - np.pi / 2.0
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    return np.einsum("tij,tkj->tki", R, h).reshape(len(h), -1)


def orientation(hand_xy: np.ndarray, is_left: bool) -> np.ndarray:
    """Wrist-to-index-MCP and wrist-to-pinky-MCP vectors plus finger-axis angles.
    Palm normal needs 3D and is skipped: our pose is 2D (impl. note 6 allows this)."""
    h = hand_xy[:, :, :2].astype(np.float32)
    w = h[:, WRIST]
    v_idx = h[:, MCP["index"]] - w
    v_pky = h[:, MCP["pinky"]] - w
    if is_left:
        v_idx = v_idx * np.array([-1.0, 1.0], np.float32)
        v_pky = v_pky * np.array([-1.0, 1.0], np.float32)
    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
    ui, up = unit(v_idx), unit(v_pky)
    ang = np.arctan2(ui[:, 1], ui[:, 0])
    spread = np.arccos(np.clip((ui * up).sum(-1), -1, 1))
    finger_ang = []
    for t in TIPS:
        v = unit(h[:, t] - w)
        finger_ang.append(np.arctan2(v[:, 1], v[:, 0]))
    return np.concatenate([ui, up, np.cos(ang)[:, None], np.sin(ang)[:, None],
                           spread[:, None], np.stack(finger_ang, -1)], -1)


def location(pose_n: np.ndarray, lh_n: np.ndarray, rh_n: np.ndarray) -> np.ndarray:
    """Wrist positions relative to shoulder midpoint, nose and torso centre, plus
    hand-to-face, hand-to-torso and inter-hand distances (impl. note 6)."""
    nose = pose_n[:, NOSE]
    torso = (pose_n[:, L_SH] + pose_n[:, R_SH]) / 2.0
    lw, rw = pose_n[:, L_WR], pose_n[:, R_WR]
    lc, rc = lh_n.mean(1), rh_n.mean(1)
    d = lambda a, b: np.linalg.norm(a - b, axis=-1, keepdims=True)
    return np.concatenate([
        lw, rw, lw - nose, rw - nose, lw - torso, rw - torso,
        d(lc, nose), d(rc, nose), d(lc, torso), d(rc, torso), d(lc, rc),
    ], -1)


def motion(pose_n: np.ndarray, lh_n: np.ndarray, rh_n: np.ndarray, win: int) -> np.ndarray:
    """Windowed velocity, acceleration, speed, direction, curvature, vertical
    displacement and change in inter-hand distance (impl. note 6)."""
    out = []
    T = len(pose_n)
    lw, rw = pose_n[:, L_WR], pose_n[:, R_WR]
    lc, rc = lh_n.mean(1), rh_n.mean(1)
    inter = np.linalg.norm(lc - rc, axis=-1)
    for s in range(0, max(T - win, 1), max(win // 2, 1)):
        for w in (lw[s:s + win], rw[s:s + win]):
            if len(w) < 3:
                continue
            v = np.diff(w, axis=0)
            a = np.diff(v, axis=0) if len(v) > 1 else np.zeros((1, 2), np.float32)
            disp = w[-1] - w[0]
            path = np.linalg.norm(v, axis=-1).sum()
            straight = np.linalg.norm(disp) / (path + 1e-6)
            ang = np.arctan2(disp[1], disp[0])
            out.append([np.cos(ang), np.sin(ang), path, straight,
                        np.linalg.norm(v, axis=-1).mean(),
                        np.linalg.norm(a, axis=-1).mean(),
                        disp[1], inter[min(s + win - 1, T - 1)] - inter[s]])
    return np.asarray(out, np.float32) if out else np.zeros((0, 8), np.float32)


def coordination(pose_n: np.ndarray, lh_n: np.ndarray, rh_n: np.ndarray,
                 win: int) -> np.ndarray:
    """Left/right energy, one- vs two-hand activity, velocity and mirrored-motion
    correlation, symmetry, inter-hand distance (impl. note 6)."""
    out = []
    T = len(pose_n)
    lw, rw = pose_n[:, L_WR], pose_n[:, R_WR]
    vl = np.linalg.norm(np.diff(lw, axis=0, prepend=lw[:1]), axis=-1)
    vr = np.linalg.norm(np.diff(rw, axis=0, prepend=rw[:1]), axis=-1)
    inter = np.linalg.norm(lh_n.mean(1) - rh_n.mean(1), axis=-1)
    for s in range(0, max(T - win, 1), max(win // 2, 1)):
        a, b = vl[s:s + win], vr[s:s + win]
        if len(a) < 3:
            continue
        el, er = float(a.mean()), float(b.mean())
        cc = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else 0.0
        mirror = lw[s:s + win].copy(); mirror[:, 0] *= -1
        mv = np.linalg.norm(np.diff(mirror, axis=0, prepend=mirror[:1]), axis=-1)
        mc = float(np.corrcoef(mv, b)[0, 1]) if mv.std() > 1e-8 and b.std() > 1e-8 else 0.0
        out.append([el, er, abs(el - er) / (el + er + 1e-6),
                    float(el > 0.01) + float(er > 0.01),
                    np.nan_to_num(cc), np.nan_to_num(mc),
                    float(inter[s:s + win].mean()), float(inter[s:s + win].std())])
    return np.asarray(out, np.float32) if out else np.zeros((0, 8), np.float32)


def nonmanual(face: np.ndarray, pose_n: np.ndarray) -> np.ndarray:
    """Head displacement and rotation proxies, eyebrow/eye/mouth apertures and their
    temporal changes (impl. note 6)."""
    f = face[:, :, :2].astype(np.float32)
    if f.shape[1] < 11:
        return np.zeros((0, 6), np.float32)
    nose = pose_n[:, NOSE]
    head_d = np.linalg.norm(np.diff(nose, axis=0, prepend=nose[:1]), axis=-1)
    eye = np.linalg.norm(f[:, L_EYE] - f[:, R_EYE], axis=-1)
    mouth = np.linalg.norm(f[:, MOUTH_L] - f[:, MOUTH_R], axis=-1)
    energy = np.linalg.norm(np.diff(f, axis=0, prepend=f[:1]), axis=-1).mean(-1)
    roll = np.arctan2(f[:, L_EYE, 1] - f[:, R_EYE, 1],
                      f[:, L_EYE, 0] - f[:, R_EYE, 0] + 1e-6)
    return np.stack([head_d, eye, mouth, energy, roll,
                     np.diff(mouth, prepend=mouth[:1])], -1).astype(np.float32)
