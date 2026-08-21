"""Unit and leakage tests (impl. note 14).

The leakage test is the one that matters. Every claim in this project rests on the
selector never having seen a label, and that property is easy to break by accident:
one import, one convenience argument, one debugging line that peeks at gloss counts,
and the result becomes meaningless while still looking fine. Asserting it in code is
the only way it stays true as the codebase changes.

Run:  python test_pc.py --poses /path/to/poses/train
"""
from __future__ import annotations

import argparse
import ast
import glob
import inspect
import sys
from pathlib import Path

import numpy as np

FAILURES: list = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 14.1 normalization
# ---------------------------------------------------------------------------

def test_normalization(pose_files):
    from descriptors_pc import PCDescriptorConfig, normalize_pose
    print("\n14.1 normalization")
    cfg = PCDescriptorConfig()
    z = np.load(pose_files[0], allow_pickle=True)
    pn, R, mid, w = normalize_pose(z["pose"], cfg)

    shoulders = (pn[:, 11] + pn[:, 12]) / 2.0
    check("shoulder midpoint ~ 0", np.abs(shoulders).max() < 1e-4,
          f"max={np.abs(shoulders).max():.2e}")

    width = np.linalg.norm(pn[:, 11] - pn[:, 12], axis=-1)
    check("shoulder width ~ 1", np.abs(width - 1.0).max() < 1e-3,
          f"mean={width.mean():.4f}")

    axis_dy = np.abs((pn[:, 11] - pn[:, 12])[:, 1])
    check("shoulder axis horizontal after rotation", axis_dy.max() < 1e-4,
          f"max|dy|={axis_dy.max():.2e}")

    check("coords clipped to range", np.abs(pn).max() <= cfg.clip_range + 1e-6,
          f"max={np.abs(pn).max():.2f}")

    small = np.abs(pn) < cfg.clip_range - 1e-3
    check("clipping leaves in-range values untouched", small.any())


# ---------------------------------------------------------------------------
# 14.2 budget constraint
# ---------------------------------------------------------------------------

def test_budget(pose_files):
    from acquire import STRATEGIES, clip_cost
    print("\n14.2 budget constraint")
    rng = np.random.default_rng(0)
    dur = rng.uniform(1.0, 12.0, 400)
    for B in (0.5, 2.0, 5.0):
        budget = B * 3600.0
        for name in ("random", "duration_random"):
            idx = STRATEGIES[name](dur, budget, "translation", seed=0)
            spent = sum(clip_cost(dur[i], "translation") for i in idx)
            check(f"{name} b={B}h respects budget", spent <= budget + 1e-6,
                  f"{spent/3600:.3f}/{B}h")


# ---------------------------------------------------------------------------
# 14.3 no label leakage
# ---------------------------------------------------------------------------

LABEL_TOKENS = ("gloss", "translation", "orth", "text_vocab", "gloss_vocab")
SELECTOR_MODULES = ("acquire", "posecover", "featurizer_pc", "descriptors_pc")


def test_no_label_leakage(code_dir: Path):
    """Static check: selector modules must not READ label fields.

    Reads the source rather than trusting runtime behaviour, because a leak on a code
    path that happens not to fire would pass a runtime test and still invalidate every
    number.

    The check targets label *access*, not the words themselves. Naming an annotation
    type is legitimate and unavoidable: the cost model has to say that a gloss costs
    40x realtime and a translation 6x. What must never appear is reading the label
    off a clip (`c.gloss`) or out of a record (`row["orth"]`). So attribute access and
    subscripting are flagged; dictionary keys in a plain literal are not.
    """
    print("\n14.3 no label leakage")
    for mod in SELECTOR_MODULES:
        p = code_dir / f"{mod}.py"
        if not p.exists():
            check(f"{mod} present", False, "missing")
            continue
        tree = ast.parse(p.read_text())

        # keys of module-level dict literals are annotation-type names, not reads
        allowed = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for k in node.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        allowed.add(id(k))

        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in LABEL_TOKENS:
                hits.append(f"read .{node.attr} (line {node.lineno})")
            if isinstance(node, ast.Subscript):
                s = node.slice
                if (isinstance(s, ast.Constant) and isinstance(s.value, str)
                        and s.value in LABEL_TOKENS and id(s) not in allowed):
                    hits.append(f"read ['{s.value}'] (line {node.lineno})")
        check(f"{mod} never reads labels", not hits,
              "; ".join(hits[:3]) if hits else "")

    # runtime check: the selector signature must not accept labels
    from posecover import select_posecover
    params = set(inspect.signature(select_posecover).parameters)
    check("select_posecover takes no label argument",
          not (params & set(LABEL_TOKENS)), ",".join(sorted(params & set(LABEL_TOKENS))))


# ---------------------------------------------------------------------------
# 14.4 determinism
# ---------------------------------------------------------------------------

def test_determinism(pose_files):
    from acquire import STRATEGIES
    from posecover import ABLATIONS, select_posecover
    print("\n14.4 determinism")
    rng = np.random.default_rng(0)
    dur = rng.uniform(1.0, 12.0, 200)
    feats = [{f"H:{rng.integers(0, 64)}": float(rng.integers(1, 9))} for _ in range(200)]
    emb = rng.normal(size=(200, 16)); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = rng.uniform(0.5, 1.0, 200)
    src = [f"S{i%9}" for i in range(200)]

    a = select_posecover(dur, 3600.0, "translation", artic_feats=feats,
                         embeddings=emb, quality=q, sources=src, cfg=ABLATIONS["full"])
    b = select_posecover(dur, 3600.0, "translation", artic_feats=feats,
                         embeddings=emb, quality=q, sources=src, cfg=ABLATIONS["full"])
    check("deterministic selector returns identical ids", a == b, f"n={len(a)}")

    r1 = STRATEGIES["random"](dur, 3600.0, "translation", seed=7)
    r2 = STRATEGIES["random"](dur, 3600.0, "translation", seed=7)
    r3 = STRATEGIES["random"](dur, 3600.0, "translation", seed=8)
    check("same seed gives same subset", r1 == r2)
    check("different seed gives different subset", r1 != r3)


# ---------------------------------------------------------------------------
# 14.5 CTC lengths
# ---------------------------------------------------------------------------

def test_ctc_lengths():
    import torch
    from data_phoenix import FEATURE_DIM
    from pose_ctc import CTCConfig, PoseCTC
    print("\n14.5 CTC lengths")
    m = PoseCTC(CTCConfig(feature_dim=FEATURE_DIM), 50)
    B, T = 4, 120
    x = torch.randn(B, T, FEATURE_DIM)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 60:] = False
    h, in_len = m.encode(x, mask)
    check("encoder downsamples by conv stride product",
          h.size(1) == T // m.downsample, f"T={T} -> {h.size(1)}, /{m.downsample}")
    check("output lengths <= encoded T", bool((in_len <= h.size(1)).all()))
    check("masked row has shorter length", int(in_len[1]) < int(in_len[0]),
          f"{int(in_len[1])} < {int(in_len[0])}")

    # a row violating T' >= L must be dropped, not produce inf
    batch = {"x": x, "x_mask": mask,
             "gloss": torch.randint(4, 50, (B, 40)),
             "gloss_len": torch.tensor([5, 40, 5, 5])}
    out = m(batch)
    check("loss finite when a row violates T'>=L", bool(torch.isfinite(out["loss"])),
          f"dropped={out['n_dropped']}")


# ---------------------------------------------------------------------------
# metrics sanity
# ---------------------------------------------------------------------------

def test_metrics():
    from metrics_pc import aulc, rare_gloss_recall, source_entropy, vocab_metrics
    from pose_ctc import corpus_wer, levenshtein
    print("\nmetrics")
    check("levenshtein known case", levenshtein(list("abc"), list("axcd")) == (1, 0, 1))
    check("perfect hypothesis gives WER 0",
          corpus_wer([["A", "B"]], [["A", "B"]])["wer"] == 0.0)
    check("empty hypothesis gives WER 100",
          corpus_wer([["A", "B"]], [[]])["wer"] == 100.0)
    # Eq. 30 integrates from the first budget, so a perfect system on [1,2,3] scores 2/3
    check("AULC perfect system, grid [1,2,3]", abs(aulc([1, 2, 3], [0, 0, 0]) - 2/3) < 1e-6,
          f"{aulc([1, 2, 3], [0, 0, 0]):.4f}")
    check("AULC perfect system, grid starting at 0", abs(aulc([0, 1, 2], [0, 0, 0]) - 1.0) < 1e-6)
    check("AULC worst system is 0", abs(aulc([0, 1, 2], [100, 100, 100])) < 1e-9)
    check("source entropy: uniform is 1",
          abs(source_entropy(["a", "b", "c", "a", "b", "c"]) - 1.0) < 1e-6)
    check("source entropy: single source is 0", source_entropy(["a"] * 5) == 0.0)
    v = vocab_metrics([["A"]], [["A"], ["B"], ["B"]], [["A"], ["B"]])
    check("gloss coverage 1 of 2 types", abs(v["gloss_coverage"] - 0.5) < 1e-9,
          f"{v['gloss_coverage']:.2f}")
    check("dev OOV counts unbought tokens", abs(v["dev_oov_token_rate"] - 50.0) < 1e-6,
          f"{v['dev_oov_token_rate']:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", default="")
    ap.add_argument("--code-dir", default=str(Path(__file__).parent))
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.poses}/*.npz"))[:5] if args.poses else []
    print("=" * 62)
    print("PoseCover unit and leakage tests (impl. note 14)")
    print("=" * 62)

    if files:
        test_normalization([Path(f) for f in files])
        test_budget(files)
    else:
        print("\n(skipping pose-dependent tests: no --poses given)")

    test_no_label_leakage(Path(args.code_dir))
    test_determinism(files)
    test_ctc_lengths()
    test_metrics()

    print("\n" + "=" * 62)
    print(f"{'ALL PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    print("=" * 62)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
