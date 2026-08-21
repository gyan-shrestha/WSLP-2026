"""Six-group articulatory featurizer with codebook fitting per impl. note 6.2.

Fitting follows the spec: median/IQR standardization, at most 200 descriptors sampled
per clip and 200,000 across the pool, MiniBatch k-means with batch 4096, k-means++
init, seed 0, at least 100 mini-batch updates.

Median/IQR rather than mean/std is not a detail. Pose descriptors carry heavy tails
from tracking failure, and a handful of frames where a hand jumps across the image
would otherwise dominate the scaling and pull the codebook towards artifacts.

Undetected hands are excluded before any descriptor is computed. Feeding them in as
zeros mints a codeword that means "hand not found", which then looks like a rare
articulation to the coverage objective and gets deliberately bought.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from descriptors_pc import (GROUPS, PCDescriptorConfig, coordination,
                            hand_configuration, location, motion, nonmanual,
                            normalize_hand_to_body, normalize_pose, orientation,
                            stable_frames)

STABLE_GROUPS = ("H", "O", "L")      # 6.1: held configurations only
ALL_FRAME_GROUPS = ("M", "C", "N")   # transitions are the signal


class PCFeaturizer:
    def __init__(self, cfg: PCDescriptorConfig = PCDescriptorConfig()):
        self.cfg = cfg
        self.codebooks: Dict[str, np.ndarray] = {}
        self.scalers: Dict[str, tuple] = {}
        self.hand_pca = None
        self.face_ok_frac = 1.0
        self.fitted = False

    # -- descriptor extraction ---------------------------------------------
    def clip_descriptors(self, npz_path: Path) -> Dict[str, np.ndarray]:
        z = np.load(npz_path, allow_pickle=True)
        pose, lh, rh, face, ok = z["pose"], z["lh"], z["rh"], z["face"], z["ok"]
        cfg = self.cfg

        pose_n, R, mid, width = normalize_pose(pose, cfg)
        lh_n = normalize_hand_to_body(lh, R, mid, width, cfg)
        rh_n = normalize_hand_to_body(rh, R, mid, width, cfg)

        out: Dict[str, List[np.ndarray]] = {g: [] for g in GROUPS}

        for hand_raw, hand_n, ok_col, wrist_i, is_left in (
                (rh, rh_n, 2, 16, False), (lh, lh_n, 1, 15, True)):
            det = ok[:, ok_col]
            if not det.any():
                continue
            # 6.1 stable frames, computed on detected frames only
            stab = stable_frames(pose_n[:, wrist_i], hand_n.mean(1),
                                 cfg.stable_percentile)
            sel = det & stab
            if sel.any():
                out["H"].append(hand_configuration(hand_raw[sel], is_left))
                out["O"].append(orientation(hand_raw[sel], is_left))

        det_both = ok[:, 1] | ok[:, 2]
        stab_all = stable_frames(pose_n[:, 16], rh_n.mean(1), cfg.stable_percentile)
        sel_l = det_both & stab_all
        if sel_l.any():
            out["L"].append(location(pose_n[sel_l], lh_n[sel_l], rh_n[sel_l]))

        out["M"].append(motion(pose_n, lh_n, rh_n, cfg.motion_window))
        out["C"].append(coordination(pose_n, lh_n, rh_n, cfg.coord_window))
        if ok[:, 3].mean() >= cfg.min_face_frac:
            out["N"].append(nonmanual(face, pose_n))

        return {g: (np.concatenate(v, 0) if v else np.zeros((0, 1), np.float32))
                for g, v in out.items()}

    # -- 6.2 codebook fitting ----------------------------------------------
    def fit(self, pose_paths: Sequence[Path], verbose: bool = True):
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA

        rng = np.random.default_rng(self.cfg.seed)
        banks: Dict[str, List[np.ndarray]] = {g: [] for g in GROUPS}
        face_ok = []

        for p in pose_paths:
            d = self.clip_descriptors(p)
            face_ok.append(len(d["N"]) > 0)
            for g in GROUPS:
                x = d[g]
                if len(x) == 0:
                    continue
                if len(x) > self.cfg.max_desc_per_clip:      # <=200 per clip
                    x = x[rng.choice(len(x), self.cfg.max_desc_per_clip, replace=False)]
                banks[g].append(x)

        self.face_ok_frac = float(np.mean(face_ok)) if face_ok else 0.0

        for g in GROUPS:
            if not banks[g]:
                continue
            X = np.concatenate(banks[g], 0)
            if len(X) > self.cfg.max_desc_pool:              # <=200k pool-wide
                X = X[rng.choice(len(X), self.cfg.max_desc_pool, replace=False)]

            if g == "H":                                     # 6: reduce to 20 PCA dims
                self.hand_pca = PCA(n_components=min(self.cfg.hand_pca_dim, X.shape[1]),
                                    random_state=self.cfg.seed).fit(X)
                X = self.hand_pca.transform(X)

            med = np.median(X, axis=0)
            q75, q25 = np.percentile(X, [75, 25], axis=0)
            iqr = np.maximum(q75 - q25, 1e-6)                # 6.2 median/IQR
            self.scalers[g] = (med, iqr)
            Xs = (X - med) / iqr

            k = min(self.cfg.n_codewords[g], max(len(Xs) // 10, 2))
            km = MiniBatchKMeans(n_clusters=k, batch_size=self.cfg.kmeans_batch,
                                 init="k-means++", random_state=self.cfg.seed,
                                 n_init=3, max_iter=200,
                                 max_no_improvement=None).fit(Xs)
            self.codebooks[g] = km.cluster_centers_
            if verbose:
                print(f"  {g}: {len(Xs):7d} descriptors -> {k:3d} codewords "
                      f"(spec {self.cfg.n_codewords[g]})", flush=True)

        if self.face_ok_frac < self.cfg.min_face_frac and verbose:
            print(f"  N disabled: face landmarks in only {self.face_ok_frac:.0%} "
                  f"of clips (< {self.cfg.min_face_frac:.0%})", flush=True)
        self.fitted = True
        return self

    def _assign(self, g: str, X: np.ndarray) -> np.ndarray:
        if g not in self.codebooks or len(X) == 0:
            return np.zeros(0, int)
        if g == "H" and self.hand_pca is not None:
            X = self.hand_pca.transform(X)
        med, iqr = self.scalers[g]
        Xs = (X - med) / iqr
        C = self.codebooks[g]
        return ((Xs[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)

    def __call__(self, npz_path: Path) -> Dict[str, float]:
        """Clip -> {group:codeword -> count}, the n_igk of Eq. 8 before capping."""
        d = self.clip_descriptors(npz_path)
        feats: Dict[str, float] = {}
        for g in GROUPS:
            for k in self._assign(g, d[g]):
                key = f"{g}:{int(k)}"
                feats[key] = feats.get(key, 0.0) + 1.0
        return feats


def fit_pc_featurizer(pose_paths: Sequence[Path], n_fit: int = 1200,
                      cfg: Optional[PCDescriptorConfig] = None, seed: int = 0):
    cfg = cfg or PCDescriptorConfig(seed=seed)
    rng = np.random.default_rng(seed)
    sel = (list(pose_paths) if len(pose_paths) <= n_fit
           else [pose_paths[i] for i in rng.choice(len(pose_paths), n_fit, replace=False)])
    print(f"fitting six-group codebooks on {len(sel)} clips", flush=True)
    return PCFeaturizer(cfg).fit(sel)


def pc_features(feat: PCFeaturizer, pose_paths: Sequence[Path]) -> List[Dict[str, float]]:
    return [feat(p) for p in pose_paths]
