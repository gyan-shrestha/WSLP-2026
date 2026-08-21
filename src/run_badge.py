"""Driver for BADGE, which needs the same two-pass structure as entropy AL.

Supports both corpora for the same reason: the baseline suite should be identical on
each, so that a difference between corpora is a difference in the finding rather than
in which methods were tried.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from badge import run_badge
from data_isharah import fill_durations as ish_fill
from data_isharah import load_split as ish_load
from data_phoenix import fill_durations, load_split

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True,
                help="PHOENIX14T root, or the Isharah annotations directory")
ap.add_argument("--poses", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--budgets-hours", type=float, nargs="+", default=[2.8, 5.5, 11.0])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--max-epochs", type=int, default=60)
ap.add_argument("--corpus", default="phoenix", choices=["phoenix", "isharah"])
ap.add_argument("--isharah-split", default="SI", choices=["SI", "US"])
a = ap.parse_args()

root, poses = Path(a.root), Path(a.poses)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

if a.corpus == "isharah":
    train = ish_load(poses, root, "train", a.isharah_split)
    dev = ish_load(poses, root, "dev", a.isharah_split)
    ish_fill(train); ish_fill(dev)
else:
    train = load_split(root, poses, "train")
    dev = load_split(root, poses, "dev")
    fill_durations(train); fill_durations(dev)

dur = np.array([c.duration_s for c in train])
print(f"corpus={a.corpus} pool={len(train)} clips {dur.sum()/3600:.2f}h "
      f"dev={len(dev)}", flush=True)

cache: dict = {}
for seed in a.seeds:
    for b in a.budgets_hours:
        tag = f"ctc_translation_badge_b{b}_s{seed}"
        fp = out / f"{tag}.json"
        if fp.exists():
            print(f"  [have] {tag}", flush=True)
            continue
        print(f"\n=== {tag} ===", flush=True)
        r = run_badge(train, dev, dur, b * 3600.0, "translation", seed=seed,
                      max_epochs=a.max_epochs, cache=cache, log_prefix="  ")
        r = {k: v for k, v in r.items() if not k.startswith("_")}
        r.update(budget_hours=b, seed=seed, annotation="translation", corpus=a.corpus)
        fp.write_text(json.dumps(r, indent=2))
        print(f"  -> WER {r['dev']['wer']:.2f}  ({r['total_minutes']:.1f} min)",
              flush=True)
print("\nDONE", flush=True)
