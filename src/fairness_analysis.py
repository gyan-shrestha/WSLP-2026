"""Does pose estimation failure translate into worse translation quality?

Detection rates vary sharply by signer: on Isharah the dominant hand is found in 97%
of frames for one signer and 61% for another, with 500 to 2000 clips per signer, so
the spread is not a small-sample artifact. That is a measurement fact. The question
that matters is whether it causes harm.

This script joins two things we already compute:
  per-signer landmark detection rate   (from the pose npz `ok` masks)
  per-signer dev BLEU                  (recorded by train.py for every model)

and tests whether they are related. A positive correlation means signers the pose
estimator tracks poorly also receive worse translations, i.e. the upstream failure
propagates into user-facing quality rather than being absorbed by the model. That is
a fairness result with a concrete harm attached, not just an input-quality statistic.

Reported with rank correlation (Spearman) rather than Pearson: with nine signers on
PHOENIX14T the relationship need not be linear and a single outlier would dominate r.
Confidence intervals come from bootstrap over signers, since that is the unit of
analysis and there are few of them.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def detection_by_signer(pose_dir: Path) -> dict:
    by = defaultdict(list)
    for f in sorted(glob.glob(f"{pose_dir}/*.npz")):
        z = np.load(f, allow_pickle=True)
        by[json.loads(str(z["meta"]))["signer"]].append(z["ok"].mean(0))
    return {s: np.array(v).mean(0) for s, v in by.items()}


def bleu_by_signer(runs_dir: Path) -> dict:
    """Average per-signer dev BLEU across every completed run.

    Averaging over runs rather than picking one is deliberate: a single model's
    per-signer scores are noisy, but the *ordering* of signers is driven by input
    quality and should persist across budgets, strategies and seeds. If it does not
    persist, that is itself the answer.
    """
    acc = defaultdict(list)
    n_runs = 0
    for f in sorted(glob.glob(f"{runs_dir}/*.json")):
        d = json.load(open(f))
        ps = d.get("per_signer_bleu") or {}
        if not ps:
            continue
        n_runs += 1
        for s, b in ps.items():
            acc[s].append(b)
    return {s: float(np.mean(v)) for s, v in acc.items()}, n_runs


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def bootstrap_ci(x, y, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(idx.tolist())) < 3:
            continue
        vals.append(spearman(np.asarray(x)[idx], np.asarray(y)[idx]))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def permutation_p(x, y, n_perm=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = spearman(x, y)
    y = np.asarray(y)
    cnt = sum(abs(spearman(x, rng.permutation(y))) >= abs(obs) for _ in range(n_perm))
    return obs, (cnt + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    det = detection_by_signer(Path(args.poses))
    bleu, n_runs = bleu_by_signer(Path(args.runs))
    shared = sorted(set(det) & set(bleu))
    if len(shared) < 4:
        raise SystemExit(f"only {len(shared)} signers in common; need >=4")

    print(f"signers={len(shared)}  runs averaged={n_runs}\n")
    print(f"{'signer':12s} {'pose':>6s} {'lhand':>7s} {'rhand':>7s} {'face':>7s} {'BLEU':>7s}")
    rows = []
    for s in shared:
        d = det[s]
        print(f"{s:12s} {d[0]:6.3f} {d[1]:7.3f} {d[2]:7.3f} {d[3]:7.3f} {bleu[s]:7.2f}")
        rows.append((s, d[1], d[2], d[3], bleu[s]))

    out = {"n_signers": len(shared), "n_runs": n_runs,
           "per_signer": {s: {"lhand": r1, "rhand": r2, "face": r3, "bleu": b}
                          for s, r1, r2, r3, b in rows}}

    y = [r[4] for r in rows]
    print()
    for name, col in (("left hand", 1), ("right hand", 2),
                      ("face", 3), ("both hands (mean)", None)):
        x = ([(r[1] + r[2]) / 2 for r in rows] if col is None else [r[col] for r in rows])
        rho, p = permutation_p(x, y)
        lo, hi = bootstrap_ci(x, y)
        print(f"{name:20s} detection vs BLEU:  rho={rho:+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}")
        out[f"rho_{name.replace(' ', '_')}"] = {"rho": rho, "ci": [lo, hi], "p": p}

    # A harm claim needs both a real effect size and evidence it is not noise. Keying
    # on rho alone is how an underpowered correlation over nine signers turns into a
    # sentence a reviewer can falsify.
    h = out["rho_both_hands_(mean)"]
    strong = h["rho"] > 0.3 and h["p"] < 0.05 and h["ci"][0] > 0
    print("\nINTERPRETATION:", (
        "signers tracked worse by pose estimation also receive worse translations; "
        "the upstream failure propagates to user-facing quality"
        if strong else
        f"UNDERPOWERED (rho={h['rho']:+.3f}, p={h['p']:.3f}, CI includes "
        f"{'zero' if h['ci'][0] <= 0 else 'small effects'}, n={len(shared)} signers). "
        "Report the input-quality disparity, which is directly measured, but do NOT "
        "claim it propagates to translation quality on this evidence."))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
