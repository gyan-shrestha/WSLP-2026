"""Do the discovered articulatory clusters track articulation, or signer identity?

Any codebook fit to pose will produce clusters. Whether those clusters separate
handshapes, rather than the people making them, is a separate question and it is
rarely tested. The failure mode is concrete: MiniBatchKMeans over handshape
descriptors will happily converge on clusters that separate *signers* by hand size,
skin tone, camera distance or seating position. Such clusters still yield a coverage
curve, and the curve still looks reasonable, so the problem is invisible downstream.

This script makes the claim falsifiable. It is the check behind the caution in the
PoseCover limitations that pose clusters should not be presented as automatically
discovered phonemes: rather than asserting the clusters are linguistic or asserting
they are not, it measures how much of the assignment signer identity explains.

Three tests, all quantitative, none requiring gold articulatory labels:

  1. SIGNER LEAKAGE.  Normalized mutual information between cluster id and signer id,
     against a permutation null. A featurizer that has learned handshape should score
     near the null; one that has learned "who is signing" scores far above it.

  2. TEMPORAL COHERENCE.  Adjacent frames within a clip are usually the same or a
     neighboring handshape. If cluster ids flip randomly frame to frame, the codebook
     is fitting noise rather than articulatory state.

  3. CHANNEL INDEPENDENCE.  HS and LOC should carry different information. If the
     handshape codebook is really encoding where the hand is in the frame (a proxy
     for camera geometry and body position), HS and LOC will be highly redundant.

Exit status is nonzero when the gate fails, so this can be wired into a pipeline.

Usage:
    python cluster_gate.py --poses /path/to/poses/train --n-clips 800
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
from descriptors import ArticulatoryConfig, ArticulatoryFeaturizer  # noqa: E402


# ---------------------------------------------------------------------------
# information-theoretic helpers
# ---------------------------------------------------------------------------

def entropy(labels) -> float:
    c = np.array(list(Counter(labels).values()), dtype=np.float64)
    p = c / c.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def mutual_information(a, b) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    joint = Counter(zip(a, b))
    ca, cb = Counter(a), Counter(b)
    mi = 0.0
    for (x, y), nxy in joint.items():
        pxy = nxy / n
        mi += pxy * np.log(pxy / ((ca[x] / n) * (cb[y] / n)) + 1e-12)
    return float(mi)


def normalized_mi(a, b) -> float:
    """NMI in [0, 1]. 0 = independent, 1 = cluster id determines signer id."""
    ha, hb = entropy(a), entropy(b)
    if ha <= 0 or hb <= 0:
        return 0.0
    return mutual_information(a, b) / np.sqrt(ha * hb)


def permutation_null(cluster_ids, signer_ids, n_perm=200, seed=0):
    """NMI we would see from chance alone, given these cluster and signer marginals.

    Raw NMI is biased upward when clusters are many and signers are few, so comparing
    against zero is meaningless. The null distribution is the only honest reference.
    """
    rng = np.random.default_rng(seed)
    sig = np.asarray(signer_ids)
    return np.array([normalized_mi(cluster_ids, rng.permutation(sig))
                     for _ in range(n_perm)])


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_poses(pose_dir, n_clips, seed=0):
    files = sorted(glob.glob(f"{pose_dir}/*.npz"))
    if not files:
        raise SystemExit(f"no npz found under {pose_dir}")
    rng = np.random.default_rng(seed)
    if n_clips and n_clips < len(files):
        files = [files[i] for i in rng.choice(len(files), n_clips, replace=False)]
    clips = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        clips.append({
            "pose": d["pose"], "lh": d["lh"], "rh": d["rh"], "face": d["face"],
            "ok": d["ok"], "signer": meta["signer"], "clip_id": meta["clip_id"],
        })
    return clips


# ---------------------------------------------------------------------------
# per-frame cluster assignment, carrying provenance
# ---------------------------------------------------------------------------

def frame_assignments(feat: ArticulatoryFeaturizer, clips):
    """Assign every *detected* hand frame to a handshape cluster, keeping the signer
    and clip it came from. Frames where the hand was not detected are dropped , 
    including them would let detection failure masquerade as a handshape category."""
    hs_ids, signers, clip_ids, frame_ix, loc_ids = [], [], [], [], []
    lay, cfg = feat.layout, feat.cfg
    for c in clips:
        norm, _ = feat._normalize_body(c["pose"])
        gx, gy = cfg.space_grid
        for hand_key, ok_col, wrist in (("rh", 2, lay.right_wrist),
                                        ("lh", 1, lay.left_wrist)):
            hand = c[hand_key]
            det = c["ok"][:, ok_col]
            if not det.any():
                continue
            desc = feat._handshape_descriptor(hand)
            ids = feat._assign(desc, feat.hs_codebook)
            p = norm[:, wrist, :]
            cx = np.clip(((p[:, 0] + 1.5) / 3.0 * gx).astype(int), 0, gx - 1)
            cy = np.clip(((p[:, 1] + 1.0) / 2.5 * gy).astype(int), 0, gy - 1)
            for t in np.nonzero(det)[0]:
                hs_ids.append(int(ids[t]))
                loc_ids.append(int(cx[t] * gy + cy[t]))
                signers.append(c["signer"])
                clip_ids.append(c["clip_id"])
                frame_ix.append(int(t))
    return (np.array(hs_ids), np.array(signers, dtype=object),
            np.array(clip_ids, dtype=object), np.array(frame_ix), np.array(loc_ids))


# ---------------------------------------------------------------------------
# the three tests
# ---------------------------------------------------------------------------

def test_signer_leakage(hs_ids, signers, seed=0):
    obs = normalized_mi(hs_ids, signers)
    null = permutation_null(hs_ids, signers, seed=seed)
    mu, sd = null.mean(), null.std() + 1e-12
    z = (obs - mu) / sd
    excess = obs - mu
    return {
        "observed_nmi": obs, "null_mean": float(mu), "null_std": float(sd),
        "z_score": float(z), "excess_nmi": float(excess),
        # Excess NMI is the interpretable quantity: how much of the cluster
        # assignment is explained by signer identity beyond chance.
        "verdict": "PASS" if excess < 0.05 else ("BORDERLINE" if excess < 0.15 else "FAIL"),
    }


def test_temporal_coherence(hs_ids, clip_ids, frame_ix):
    """Fraction of adjacent frame pairs (same clip, consecutive) sharing a cluster,
    compared against the rate expected if ids were drawn from the marginal."""
    by_clip = defaultdict(list)
    for h, c, t in zip(hs_ids, clip_ids, frame_ix):
        by_clip[c].append((t, h))
    same = total = 0
    for seq in by_clip.values():
        seq.sort()
        for (t1, h1), (t2, h2) in zip(seq, seq[1:]):
            if t2 == t1 + 1:
                total += 1
                same += (h1 == h2)
    counts = np.array(list(Counter(hs_ids).values()), np.float64)
    p = counts / counts.sum()
    chance = float((p ** 2).sum())
    obs = same / max(total, 1)
    return {
        "adjacent_pairs": total, "observed_same": obs, "chance_same": chance,
        "lift": obs / (chance + 1e-12),
        "verdict": "PASS" if obs > 3 * chance else ("BORDERLINE" if obs > 1.5 * chance else "FAIL"),
    }


def test_channel_independence(hs_ids, loc_ids):
    nmi = normalized_mi(hs_ids, loc_ids)
    return {
        "hs_loc_nmi": nmi,
        "verdict": "PASS" if nmi < 0.30 else ("BORDERLINE" if nmi < 0.50 else "FAIL"),
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True, help="dir of per-clip npz")
    ap.add_argument("--n-clips", type=int, default=800)
    ap.add_argument("--n-handshape-clusters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    print(f"loading up to {args.n_clips} clips from {args.poses}", flush=True)
    clips = load_poses(args.poses, args.n_clips, args.seed)
    signers = Counter(c["signer"] for c in clips)
    print(f"  {len(clips)} clips, {len(signers)} signers: {dict(signers)}")

    cfg = ArticulatoryConfig(n_handshape_clusters=args.n_handshape_clusters, seed=args.seed)
    feat = ArticulatoryFeaturizer(cfg)
    print("fitting codebooks on real pose ...", flush=True)
    feat.fit([c["pose"] for c in clips], [(c["lh"], c["rh"]) for c in clips])
    print(f"  handshape codebook: {feat.hs_codebook.shape}")
    print(f"  movement codebook:  {feat.mv_codebook.shape}")

    hs, sg, cl, fi, lc = frame_assignments(feat, clips)
    print(f"  {len(hs)} detected hand-frames, {len(set(hs.tolist()))} clusters used")
    if len(hs) < 1000:
        print("WARNING: very few detected hand-frames; results are not trustworthy")

    results = {
        "n_clips": len(clips), "n_signers": len(signers), "n_hand_frames": int(len(hs)),
        "signer_leakage": test_signer_leakage(hs, sg, args.seed),
        "temporal_coherence": test_temporal_coherence(hs, cl, fi),
        "channel_independence": test_channel_independence(hs, lc),
    }

    print("\n" + "=" * 68)
    print("WEEK-2 GATE")
    print("=" * 68)
    for name, r in results.items():
        if not isinstance(r, dict):
            continue
        print(f"\n{name}   -> {r['verdict']}")
        for k, v in r.items():
            if k != "verdict":
                print(f"    {k:20s} {v:.4f}" if isinstance(v, float) else f"    {k:20s} {v}")

    verdicts = [r["verdict"] for r in results.values() if isinstance(r, dict)]
    overall = "FAIL" if "FAIL" in verdicts else ("BORDERLINE" if "BORDERLINE" in verdicts else "PASS")
    results["overall"] = overall
    print("\n" + "=" * 68)
    print(f"OVERALL: {overall}")
    if overall == "FAIL":
        print("""
Do not build the articulation claim on this featurizer as it stands. In order:
  1. strengthen normalization (shoulder-width scaling, torso rotation)
  2. drop absolute position from the handshape descriptor
  3. re-fit codebooks per signer-normalized descriptor
If it still fails, reframe the coverage term as "articulatory diversity" rather
than "articulatory coverage" and say so plainly in the paper.""")
    print("=" * 68)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json_out}")

    return 1 if overall == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
