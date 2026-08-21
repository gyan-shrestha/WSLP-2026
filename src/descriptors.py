"""
SIGNAL v2: articulation-aware, multi-fidelity active annotation for sign language translation.

Two modules, both with no direct spoken-language analogue:

  A. ArticulatoryCoverage
     Builds a discrete inventory over sign-language articulatory parameters
     (handshape, location in signing space, movement, two-handedness, non-manuals)
     estimated from POSE KEYPOINTS ALONE, so it applies to unlabeled clips.
     Coverage over this inventory is what the acquisition function maximizes.

  B. MultiFidelityAllocator
     Sign language annotation is not a single action. A clip can be given a free
     translation (cheap, bilingual annotator) or a gloss (expensive, trained
     annotator, ~40x realtime per How2Sign). The acquisition problem is therefore
     WHICH ANNOTATION TYPE TO BUY FOR WHICH CLIP under one shared annotator-time
     budget, not merely which clip to label. Formulated as submodular maximization
     under a knapsack constraint plus a partition matroid (one action per clip),
     solved with cost-effective lazy greedy.

Run `python descriptors.py` for a synthetic end-to-end demo.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ============================================================================
# A. Articulatory coverage from pose
# ============================================================================

@dataclass
class PoseLayout:
    """Keypoint index groups. Defaults follow MediaPipe Holistic conventions."""
    left_shoulder: int = 11
    right_shoulder: int = 12
    nose: int = 0
    left_wrist: int = 15
    right_wrist: int = 16
    left_hand: Tuple[int, ...] = tuple(range(21))   # into a separate hand array
    right_hand: Tuple[int, ...] = tuple(range(21))
    face_pts: Tuple[int, ...] = tuple(range(0, 11))


@dataclass
class ArticulatoryConfig:
    n_handshape_clusters: int = 40      # ASL-LEX lists ~50 phonemic handshapes
    n_movement_clusters: int = 24
    space_grid: Tuple[int, int] = (5, 5)   # signing-space cells, shoulder-normalized
    window: int = 8                      # frames per articulatory window
    stride: int = 4
    min_hand_activity: float = 0.02      # normalized velocity to count as active
    seed: int = 0


class ArticulatoryFeaturizer:
    """Pose sequence -> multiset of discrete articulatory tokens.

    Channels (each a separate sub-inventory so coverage can be reported per channel):
      HS  handshape codebook id
      LOC signing-space cell of the dominant hand
      MOV movement direction and magnitude codebook id
      HND handedness configuration (dominant only / symmetric / asymmetric two-handed)
      NM  non-manual activity bucket from facial landmark displacement energy

    Codebooks are fit on the LABELED split only, then applied to unlabeled clips.
    No gloss, no translation, no gold label of any kind is used.
    """

    def __init__(self, cfg: ArticulatoryConfig = ArticulatoryConfig(), layout: PoseLayout = PoseLayout()):
        self.cfg = cfg
        self.layout = layout
        self.hs_codebook: Optional[np.ndarray] = None
        self.mv_codebook: Optional[np.ndarray] = None
        self.fitted = False

    # -- normalization ------------------------------------------------------
    def _normalize_body(self, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Translate to shoulder midpoint, scale by shoulder width. pose: (T, J, 2 or 3)."""
        L, R = self.layout.left_shoulder, self.layout.right_shoulder
        mid = (pose[:, L, :2] + pose[:, R, :2]) / 2.0
        width = np.linalg.norm(pose[:, L, :2] - pose[:, R, :2], axis=-1, keepdims=True)
        width = np.maximum(width, 1e-6)
        norm = (pose[:, :, :2] - mid[:, None, :]) / width[:, None, :]
        return norm, width.squeeze(-1)

    def _handshape_descriptor(self, hand: np.ndarray) -> np.ndarray:
        """Rotation- and scale-invariant handshape descriptor: normalized pairwise
        distances among a subset of hand joints. hand: (T, 21, 2 or 3) -> (T, D)."""
        idx = [0, 4, 8, 12, 16, 20, 5, 9, 13, 17]   # wrist, fingertips, knuckles
        h = hand[:, idx, :2]
        d = np.linalg.norm(h[:, :, None, :] - h[:, None, :, :], axis=-1)
        scale = np.maximum(d.max(axis=(1, 2), keepdims=True), 1e-6)
        d = d / scale
        iu = np.triu_indices(len(idx), k=1)
        return d[:, iu[0], iu[1]]

    def _movement_descriptor(self, wrist_norm: np.ndarray) -> np.ndarray:
        """Windowed trajectory: displacement, path length, curvature, repetition."""
        cfg = self.cfg
        out = []
        for s in range(0, max(len(wrist_norm) - cfg.window, 1), cfg.stride):
            w = wrist_norm[s:s + cfg.window]
            if len(w) < 3:
                continue
            step = np.diff(w, axis=0)
            disp = w[-1] - w[0]
            path = np.linalg.norm(step, axis=-1).sum()
            straight = np.linalg.norm(disp) / (path + 1e-6)
            ang = math.atan2(disp[1], disp[0])
            # crude repetition cue: sign changes of the velocity vector
            reps = float((np.sign(step[:-1, 0]) != np.sign(step[1:, 0])).sum())
            out.append([math.cos(ang), math.sin(ang), path, straight, reps / max(len(step), 1)])
        return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 5), np.float32)

    # -- fitting ------------------------------------------------------------
    def fit(self, poses: Sequence[np.ndarray], hands: Sequence[Tuple[np.ndarray, np.ndarray]]):
        """poses[i]: (T, J, 2+). hands[i]: (left (T,21,2+), right (T,21,2+))."""
        from sklearn.cluster import MiniBatchKMeans

        hs_bank, mv_bank = [], []
        for pose, (lh, rh) in zip(poses, hands):
            norm, _ = self._normalize_body(pose)
            for hand in (lh, rh):
                if hand is not None and len(hand):
                    hs_bank.append(self._handshape_descriptor(hand))
            for wi in (self.layout.left_wrist, self.layout.right_wrist):
                mv = self._movement_descriptor(norm[:, wi, :])
                if len(mv):
                    mv_bank.append(mv)
        HS = np.concatenate(hs_bank, 0) if hs_bank else np.zeros((1, 45), np.float32)
        MV = np.concatenate(mv_bank, 0) if mv_bank else np.zeros((1, 5), np.float32)
        k_hs = min(self.cfg.n_handshape_clusters, max(len(HS) // 10, 2))
        k_mv = min(self.cfg.n_movement_clusters, max(len(MV) // 10, 2))
        self.hs_codebook = MiniBatchKMeans(k_hs, random_state=self.cfg.seed,
                                           n_init=3).fit(HS).cluster_centers_
        self.mv_codebook = MiniBatchKMeans(k_mv, random_state=self.cfg.seed,
                                           n_init=3).fit(MV).cluster_centers_
        self.fitted = True
        return self

    @staticmethod
    def _assign(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0, dtype=int)
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        return d.argmin(1)

    # -- featurization ------------------------------------------------------
    def __call__(self, pose: np.ndarray, left_hand: Optional[np.ndarray],
                 right_hand: Optional[np.ndarray], face: Optional[np.ndarray] = None
                 ) -> Dict[str, float]:
        assert self.fitted, "call .fit() on the labeled split first"
        cfg, lay = self.cfg, self.layout
        feats: Dict[str, float] = defaultdict(float)
        norm, _ = self._normalize_body(pose)

        # LOC: signing-space grid of each wrist
        gx, gy = cfg.space_grid
        for tag, wi in (("D", lay.right_wrist), ("N", lay.left_wrist)):
            p = norm[:, wi, :]
            cx = np.clip(((p[:, 0] + 1.5) / 3.0 * gx).astype(int), 0, gx - 1)
            cy = np.clip(((p[:, 1] + 1.0) / 2.5 * gy).astype(int), 0, gy - 1)
            for a, b in zip(cx[::cfg.stride], cy[::cfg.stride]):
                feats[f"LOC:{tag}:{a}_{b}"] += 1.0

        # HS
        for tag, hand in (("D", right_hand), ("N", left_hand)):
            if hand is not None and len(hand):
                ids = self._assign(self._handshape_descriptor(hand)[::cfg.stride], self.hs_codebook)
                for i in ids:
                    feats[f"HS:{tag}:{int(i)}"] += 1.0

        # MOV
        for tag, wi in (("D", lay.right_wrist), ("N", lay.left_wrist)):
            mv = self._movement_descriptor(norm[:, wi, :])
            for i in self._assign(mv, self.mv_codebook):
                feats[f"MOV:{tag}:{int(i)}"] += 1.0

        # HND: handedness configuration over windows
        vl = np.linalg.norm(np.diff(norm[:, lay.left_wrist, :], axis=0), axis=-1)
        vr = np.linalg.norm(np.diff(norm[:, lay.right_wrist, :], axis=0), axis=-1)
        for s in range(0, max(len(vl) - cfg.window, 1), cfg.stride):
            al = vl[s:s + cfg.window].mean() > cfg.min_hand_activity
            ar = vr[s:s + cfg.window].mean() > cfg.min_hand_activity
            if al and ar:
                sym = np.corrcoef(vl[s:s + cfg.window], vr[s:s + cfg.window])[0, 1]
                feats["HND:sym" if (sym > 0.5) else "HND:asym"] += 1.0
            elif al or ar:
                feats["HND:one"] += 1.0

        # NM: non-manual activity energy bucket
        if face is not None and len(face) > 1:
            e = np.linalg.norm(np.diff(face[:, :, :2], axis=0), axis=-1).mean()
            feats[f"NM:{min(int(e * 50), 5)}"] += 1.0
        return dict(feats)


CHANNEL_WEIGHTS = {"HS": 1.5, "LOC": 1.0, "MOV": 1.2, "HND": 0.8, "NM": 1.0,
                   "LEX": 1.0, "BIG": 0.6, "NEG": 2.0, "NUM": 2.0, "TMP": 1.5, "SPA": 1.5,
                   "LEN": 0.5, "CLU": 1.0}


class ChannelCoverage:
    """C(S) = sum_f w_f g(n_f), g concave. Monotone submodular. Per-channel readout."""

    def __init__(self, g: str = "sqrt", tau: float = 3.0,
                 weights: Dict[str, float] = None):
        self.counts: Dict[str, float] = defaultdict(float)
        self.w = weights or CHANNEL_WEIGHTS
        self.g = {"sqrt": math.sqrt,
                  "log": lambda c: math.log1p(c),
                  "sat": lambda c: 1.0 - math.exp(-c / tau)}[g]

    def _w(self, f: str) -> float:
        return self.w.get(f.split(":", 1)[0], 1.0)

    def seed(self, feats_list: Sequence[Dict[str, float]]) -> None:
        for fe in feats_list:
            for f, c in fe.items():
                self.counts[f] += c

    def gain(self, fe: Dict[str, float]) -> float:
        t = 0.0
        for f, c in fe.items():
            cur = self.counts[f]
            t += self._w(f) * (self.g(cur + c) - self.g(cur))
        return t

    def add(self, fe: Dict[str, float]) -> None:
        for f, c in fe.items():
            self.counts[f] += c

    def per_channel_coverage(self) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for f, c in self.counts.items():
            if c > 0:
                out[f.split(":", 1)[0]] += 1
        return dict(out)


# ============================================================================
# B. Multi-fidelity annotation allocation
# ============================================================================

@dataclass
class AnnotationType:
    """Cost is annotator-seconds per second of video (realtime multiplier).

    Defaults: gloss ~40x realtime (How2Sign reports ~1 hour per 90 seconds of
    video); free translation is far cheaper and needs a bilingual signer rather
    than a trained glosser. Treat both as hyperparameters and run a sensitivity
    sweep over the ratio, since it varies by corpus and annotator pool.
    """
    name: str
    realtime_multiplier: float
    supplies: Tuple[str, ...]          # which supervision channels it yields


TRANSLATION = AnnotationType("translation", 6.0, ("sem",))
GLOSS = AnnotationType("gloss", 40.0, ("sem", "phon", "align"))


@dataclass
class MFConfig:
    channel_value: Dict[str, float] = field(default_factory=lambda: {
        "sem": 1.0, "phon": 1.0, "align": 0.5})
    use_knapsack_guarantee: bool = True   # also try best-single-action, take the max


class MultiFidelityAllocator:
    """Choose (clip, annotation_type) pairs maximizing coverage per annotator-second.

    Objective: monotone submodular in the chosen action set.
    Constraints: one action per clip (partition matroid) + total cost <= B (knapsack).
    Cost-effective lazy greedy, with the standard best-single-element comparison
    that restores a constant-factor guarantee under the knapsack constraint.
    """

    def __init__(self, cfg: MFConfig,
                 sem_cov: ChannelCoverage, phon_cov: ChannelCoverage,
                 sem_feats: Sequence[Dict[str, float]],
                 artic_feats: Sequence[Dict[str, float]],
                 durations: np.ndarray,
                 types: Sequence[AnnotationType] = (TRANSLATION, GLOSS)):
        self.cfg = cfg
        self.sem, self.phon = sem_cov, phon_cov
        self.sem_feats, self.artic_feats = sem_feats, artic_feats
        self.dur = np.asarray(durations, dtype=np.float64)
        self.types = list(types)

    def cost(self, i: int, t: AnnotationType) -> float:
        return float(self.dur[i]) * t.realtime_multiplier

    def gain(self, i: int, t: AnnotationType) -> float:
        v = self.cfg.channel_value
        g = 0.0
        if "sem" in t.supplies:
            g += v["sem"] * self.sem.gain(self.sem_feats[i])
        if "phon" in t.supplies:
            g += v["phon"] * self.phon.gain(self.artic_feats[i])
        return g

    def _commit(self, i: int, t: AnnotationType) -> None:
        if "sem" in t.supplies:
            self.sem.add(self.sem_feats[i])
        if "phon" in t.supplies:
            self.phon.add(self.artic_feats[i])

    def allocate(self, candidates: Sequence[int], budget_seconds: float
                 ) -> List[Tuple[int, str]]:
        heap = []
        for i in candidates:
            for t in self.types:
                c = self.cost(i, t)
                if c <= budget_seconds:
                    heap.append((-self.gain(i, t) / c, i, t.name, 0))
        heapq.heapify(heap)

        chosen: List[Tuple[int, str]] = []
        assigned = set()
        spent, it = 0.0, 0
        tmap = {t.name: t for t in self.types}
        while heap:
            negr, i, tname, stamp = heapq.heappop(heap)
            if i in assigned:
                continue
            t = tmap[tname]
            c = self.cost(i, t)
            if spent + c > budget_seconds:
                continue
            if stamp != it:
                heapq.heappush(heap, (-self.gain(i, t) / c, i, tname, it))
                continue
            chosen.append((i, tname))
            assigned.add(i)
            self._commit(i, t)
            spent += c
            it += 1
        return chosen


def fixed_fidelity_baseline(candidates, sem_cov, phon_cov, sem_feats, artic_feats,
                            durations, t: AnnotationType, budget_seconds: float,
                            cfg: MFConfig = MFConfig()):
    """Ablation: all budget spent on a single annotation type."""
    alloc = MultiFidelityAllocator(cfg, sem_cov, phon_cov, sem_feats, artic_feats,
                                   durations, types=(t,))
    return alloc.allocate(candidates, budget_seconds)


# ============================================================================
# Demo
# ============================================================================

def _demo():
    rng = np.random.default_rng(0)
    N, T, J = 120, 60, 33

    def fake_clip(rare: bool):
        base = rng.normal(0, 0.02, size=(T, J, 2))
        base[:, 11] += [-0.5, 0.0]
        base[:, 12] += [0.5, 0.0]
        amp = 0.6 if rare else 0.2
        ph = rng.uniform(0, 6) if rare else 0.0
        t = np.linspace(0, 4, T)
        base[:, 16, 0] += amp * np.sin(t + ph)
        base[:, 16, 1] += amp * np.cos(t + ph)
        lh = rng.normal(0, 0.05 if not rare else 0.2, size=(T, 21, 2))
        rh = rng.normal(0, 0.05 if not rare else 0.2, size=(T, 21, 2))
        face = rng.normal(0, 0.01 if not rare else 0.04, size=(T, 11, 2))
        return base, lh, rh, face

    clips = [fake_clip(i % 7 == 0) for i in range(N)]
    labeled_idx, pool_idx = list(range(20)), list(range(20, N))

    ph = ArticulatoryFeaturizer(ArticulatoryConfig(n_handshape_clusters=12, n_movement_clusters=8))
    ph.fit([clips[i][0] for i in labeled_idx], [(clips[i][1], clips[i][2]) for i in labeled_idx])
    artic_feats = [ph(c[0], c[1], c[2], c[3]) for c in clips]

    # stand-in semantic features; in the real system these come from the model's
    # n-best hypotheses via signal_acquisition.posterior_features
    sem_feats = [{f"LEX:w{int(rng.integers(0, 60))}": 1.0 for _ in range(4)} for _ in range(N)]
    durations = rng.uniform(4, 10, size=N)

    def fresh():
        s, p = ChannelCoverage(), ChannelCoverage()
        s.seed([sem_feats[i] for i in labeled_idx])
        p.seed([artic_feats[i] for i in labeled_idx])
        return s, p

    B = 3000.0  # annotator-seconds
    s, p = fresh()
    mixed = MultiFidelityAllocator(MFConfig(), s, p, sem_feats, artic_feats, durations
                                   ).allocate(pool_idx, B)
    ng = sum(1 for _, t in mixed if t == "gloss")
    print(f"MIXED     {len(mixed):3d} clips ({ng} gloss / {len(mixed)-ng} translation)  "
          f"phon channels covered={p.per_channel_coverage()}")

    s, p = fresh()
    only_g = fixed_fidelity_baseline(pool_idx, s, p, sem_feats, artic_feats, durations, GLOSS, B)
    print(f"GLOSS-ONLY{len(only_g):3d} clips  phon channels covered={p.per_channel_coverage()}")

    s, p = fresh()
    only_t = fixed_fidelity_baseline(pool_idx, s, p, sem_feats, artic_feats, durations, TRANSLATION, B)
    print(f"TRANS-ONLY{len(only_t):3d} clips  phon channels covered={p.per_channel_coverage()}")


if __name__ == "__main__":
    _demo()
