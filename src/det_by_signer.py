"""Does hand-detection failure correlate with signer identity?

Detection sits at ~0.69 and upscaling did not fix it. If the miss rate varies
systematically by signer, the set of *detected* frames is itself signer-biased, and
any coverage measure computed over detected frames inherits that bias. That would
confound both the week-2 gate and the fairness analysis, so it is checked first.
"""
import glob, json, sys
import numpy as np
from collections import defaultdict

by = defaultdict(list)
for f in sorted(glob.glob(f"{sys.argv[1]}/*.npz")):
    z = np.load(f, allow_pickle=True)
    by[json.loads(str(z["meta"]))["signer"]].append(z["ok"].mean(0))

print(f"{'signer':12s} {'clips':>6s} {'pose':>7s} {'lhand':>7s} {'rhand':>7s} {'face':>7s}")
rows = []
for s in sorted(by):
    R = np.array(by[s])
    rows.append((s, len(R), *R.mean(0)))
    print(f"{s:12s} {len(R):6d} {R[:,0].mean():7.3f} {R[:,1].mean():7.3f} "
          f"{R[:,2].mean():7.3f} {R[:,3].mean():7.3f}")

rh = np.array([r[4] for r in rows]); n = np.array([r[1] for r in rows])
print(f"\nweighted mean rhand detection : {(rh*n).sum()/n.sum():.3f}")
print(f"spread across signers          : min={rh.min():.3f} max={rh.max():.3f} "
      f"range={rh.max()-rh.min():.3f} sd={rh.std():.3f}")
rng = rh.max() - rh.min()
print("\nVERDICT:", "signer-neutral" if rng < 0.10 else
      ("MODERATE signer-dependence - report it" if rng < 0.25 else
       "STRONG signer-dependence - coverage inherits this bias (and this is Idea B's finding)"))
