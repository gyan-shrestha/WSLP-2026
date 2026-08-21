"""Pose-corruption robustness study (impl. note 13.4, paper 6.8).

Randomly drop 10% and 20% of the *detected* hand landmarks, rerun descriptor
extraction and selection, and measure how much the purchase changes.

The question is whether a selector responds to articulation or to pose-estimator
artifacts. If deleting a tenth of the landmarks reshuffles which clips get bought,
then the objective was tracking noise, and the coverage it reports is not a property
of the signing.

This matters more here than injected noise usually does. Detection rates already vary
from 61% to 97% across signers in Isharah, which is a larger perturbation than either
corruption level and one that arrives unevenly across people rather than at random. A
selector that is fragile at 10% dropout is therefore already behaving inconsistently
across real signers in real corpora, and the fragility would be invisible in aggregate
numbers.

Only landmarks marked as detected are dropped. Corrupting already-missing frames would
change nothing and would understate the perturbation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def corrupt_clip(npz_path: Path, rate: float, rng) -> Dict[str, np.ndarray]:
    """Return arrays with `rate` of detected hand landmarks zeroed and unmarked.

    Both the coordinates and the detection mask are updated, so downstream code sees a
    genuinely worse pose estimate rather than zeros that still claim to be valid.
    """
    z = np.load(npz_path, allow_pickle=True)
    lh, rh, ok = z["lh"].copy(), z["rh"].copy(), z["ok"].copy()

    for arr, col in ((lh, 1), (rh, 2)):
        det = np.nonzero(ok[:, col])[0]
        if len(det) == 0:
            continue
        n_drop = int(round(rate * len(det)))
        if n_drop == 0:
            continue
        drop = rng.choice(det, n_drop, replace=False)
        arr[drop] = 0.0
        ok[drop, col] = False

    return {"pose": z["pose"], "lh": lh, "rh": rh, "face": z["face"], "ok": ok,
            "meta": z["meta"]}


def write_corrupted_cache(pose_paths: Sequence[Path], out_dir: Path,
                          rate: float, seed: int = 0) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for p in pose_paths:
        dst = out_dir / p.name
        if not dst.exists():
            np.savez_compressed(dst, **corrupt_clip(p, rate, rng))
        paths.append(dst)
    return paths


def duration_weighted_jaccard(a: Sequence[int], b: Sequence[int],
                              durations: np.ndarray) -> float:
    """Paper Eq. 40: overlap weighted by clip duration rather than clip count.

    Duration weighting is the right choice because the budget is spent in seconds. Two
    selections could share few clip ids while overlapping heavily in purchased time,
    and it is the time that was paid for.
    """
    sa, sb = set(int(i) for i in a), set(int(i) for i in b)
    inter = sum(durations[i] for i in sa & sb)
    union = sum(durations[i] for i in sa | sb)
    return float(inter / union) if union > 0 else 0.0


def run_corruption_study(pose_paths: Sequence[Path], durations: np.ndarray,
                         budget_s: float, annotation: str,
                         select_fn, cache_root: Path,
                         rates: Sequence[float] = (0.10, 0.20),
                         seed: int = 0) -> Dict:
    """select_fn(paths) -> selected indices. Called once on clean poses and once per
    corruption level, so the comparison isolates the effect of landmark loss."""
    from featurizer_pc import fit_pc_featurizer, pc_features

    t0 = time.time()
    print("clean selection ...", flush=True)
    clean_idx = select_fn(list(pose_paths))

    results = {"clean_n_clips": len(clean_idx), "levels": {}}
    for rate in rates:
        print(f"\ncorruption {rate:.0%} ...", flush=True)
        cdir = cache_root / f"corrupt_{int(rate*100)}"
        cpaths = write_corrupted_cache(pose_paths, cdir, rate, seed)

        # measure the perturbation actually applied, rather than assuming it landed
        det_clean = np.mean([np.load(p, allow_pickle=True)["ok"][:, 1:3].mean()
                             for p in pose_paths[:200]])
        det_corr = np.mean([np.load(p, allow_pickle=True)["ok"][:, 1:3].mean()
                            for p in cpaths[:200]])

        idx = select_fn(cpaths)
        j = duration_weighted_jaccard(clean_idx, idx, durations)
        results["levels"][f"{int(rate*100)}%"] = {
            "n_clips": len(idx),
            "duration_weighted_jaccard": j,
            "hand_detection_clean": float(det_clean),
            "hand_detection_corrupted": float(det_corr),
        }
        print(f"  overlap with clean selection: {j:.3f}   "
              f"detection {det_clean:.3f} -> {det_corr:.3f}", flush=True)

    results["minutes"] = (time.time() - t0) / 60.0
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--budget-hours", type=float, default=5.5)
    ap.add_argument("--strategies", nargs="+",
                    default=["pc_full", "pc_art_only", "pc_rep_only", "random"])
    ap.add_argument("--pool-limit", type=int, default=2000,
                    help="corrupted caches are written to disk, so the study runs on a "
                         "sample of the pool by default")
    args = ap.parse_args()

    from acquire import STRATEGIES, clip_cost
    from data_phoenix import fill_durations, load_split
    from featurizer_pc import fit_pc_featurizer, pc_features
    from posecover import (ABLATIONS, generic_embeddings, quality_score,
                           select_posecover)

    root, poses = Path(args.root), Path(args.poses)
    train = load_split(root, poses, "train")[: args.pool_limit]
    fill_durations(train)
    dur = np.array([c.duration_s for c in train])
    paths = [c.pose_path for c in train]
    srcs = [c.signer for c in train]
    budget_s = args.budget_hours * 3600.0

    out = {}
    for strat in args.strategies:
        print(f"\n{'='*60}\n{strat}\n{'='*60}", flush=True)

        def select_fn(pp, _s=strat):
            if _s == "random":
                return STRATEGIES["random"](dur, budget_s, "translation", seed=0)
            feat = fit_pc_featurizer(pp, n_fit=min(600, len(pp)))
            af = pc_features(feat, pp)
            emb = generic_embeddings(pp)
            q = np.array([quality_score(np.load(p, allow_pickle=True)["ok"]) for p in pp])
            return select_posecover(dur, budget_s, "translation", artic_feats=af,
                                    embeddings=emb, quality=q, sources=srcs,
                                    cfg=ABLATIONS[_s[3:]])

        out[strat] = run_corruption_study(paths, dur, budget_s, "translation",
                                          select_fn, Path(args.cache_root))

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")

    print("\n" + "=" * 60)
    print(f"{'strategy':16s} {'10% overlap':>12s} {'20% overlap':>12s}")
    for s, r in out.items():
        a = r["levels"].get("10%", {}).get("duration_weighted_jaccard", float("nan"))
        b = r["levels"].get("20%", {}).get("duration_weighted_jaccard", float("nan"))
        print(f"{s:16s} {a:12.3f} {b:12.3f}")
    print("\nrandom is the reference: its overlap reflects only sampling, not "
          "sensitivity to pose quality.")


if __name__ == "__main__":
    main()
