"""Duration confound control for the density analysis.

Informed selectors buy more, shorter clips than random at equal cost. If short clips
are systematically sparser in the pose embedding, then "coverage buys atypical clips"
is a restatement of the known clip-length effect rather than an independent finding.
Two checks: pool-wide correlation of kNN distance with duration, and the same
selector comparison recomputed within duration strata.
"""
import argparse, json
from pathlib import Path
import numpy as np
from density import knn_distance
from posecover import generic_embeddings

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--poses", required=True)
ap.add_argument("--corpus", default="phoenix"); ap.add_argument("--density-json", required=True)
ap.add_argument("--k", type=int, default=20)
a = ap.parse_args()

if a.corpus == "isharah":
    from data_isharah import fill_durations, load_split
    train = load_split(Path(a.poses), Path(a.root), "train", "SI")
else:
    from data_phoenix import fill_durations, load_split
    train = load_split(Path(a.root), Path(a.poses), "train")
fill_durations(train)

dur = np.array([c.duration_s for c in train])
d = knn_distance(generic_embeddings([c.pose_path for c in train]), k=a.k)
sel = json.load(open(a.density_json))["selected"]

print(f"\n{'='*70}\n{a.corpus}: is kNN distance just clip duration?\n{'='*70}")
r = np.corrcoef(dur, d)[0, 1]
rs = np.corrcoef(np.argsort(np.argsort(dur)), np.argsort(np.argsort(d)))[0, 1]
print(f"pool n={len(dur)}  pearson r = {r:+.3f}   spearman = {rs:+.3f}")
print("  (near zero => density is independent of length, finding stands)")

# duration quartiles, and mean kNN within each
qs = np.percentile(dur, [25, 50, 75])
strat = np.digitize(dur, qs)
print("\nduration quartile:  " + "".join("%12s" % f"Q{i+1}" for i in range(4)))
print("  n:                " + "".join("%12d" % (strat == i).sum() for i in range(4)))
print("  mean dur (s):     " + "".join("%12.2f" % dur[strat == i].mean() for i in range(4)))
print("  mean kNN dist:    " + "".join("%12.4f" % d[strat == i].mean() for i in range(4)))

# stratified comparison: within each quartile, how sparse is what each selector bought?
print("\nmean kNN distance WITHIN each duration quartile (removes the length effect)")
print("%-26s" % "strategy" + "".join("%12s" % f"Q{i+1}" for i in range(4)) + "%12s" % "n")
for name, idx in sorted(sel.items()):
    idx = np.array(idx)
    row = "%-26s" % name
    for i in range(4):
        m = idx[strat[idx] == i]
        row += "%12s" % "--" if len(m) < 10 else "%12.4f" % d[m].mean()
    print(row + "%12d" % len(idx))
row = "%-26s" % "(pool baseline)"
for i in range(4):
    row += "%12.4f" % d[strat == i].mean()
print(row + "%12d" % len(dur))
