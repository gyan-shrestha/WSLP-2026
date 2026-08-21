"""Why does coverage fail? Local density of what each selector purchases.

The paper establishes that articulatory coverage does not beat random, that adding
coverage weight hurts, that coverage saturates, and that even an oracle over gloss
types performs poorly. What it does not yet show is why the purchased data trains
worse. This measures that directly.

For every clip we compute mean distance to its K nearest neighbours in the pool, which
is an inverse local density: large distance means the clip sits in a sparse region and
resembles little else. We then compare the density profile of what each selector buys.

Two spaces, and the distinction is essential to reading the result.

  GENERIC pose embedding. The informative one. Articulatory coverage operates over
  discrete codeword counts, so finding that its purchases are also sparse in a
  continuous pose embedding is a genuine empirical claim: articulatory rarity coincides
  with general atypicality. There was no guarantee the two spaces would agree.

  ARTICULATORY space. Reported for completeness, but near-definitional. The objective
  explicitly up-weights rare codewords, so of course it selects clips that are rare in
  codeword space. This row should not be read as evidence.

For the same reason, facility location scoring high density is definitional rather than
a finding: it maximizes similarity to pool members, so it selects central points by
construction. The comparison that carries weight is articulatory-only against random.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def knn_distance(Z: np.ndarray, k: int = 20, block: int = 512) -> np.ndarray:
    """Mean cosine distance to the K nearest neighbours, excluding self.

    Blocked so an N x N similarity matrix is never materialised: at N = 10,000 that
    would be 800 MB and at N = 30,000 it would not fit.
    """
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    n = len(Z)
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = Z[s:e] @ Z.T                       # (b, n) cosine similarity
        np.fill_diagonal(sim[:, s:e], -np.inf)   # exclude self
        # top-k similarities -> mean distance = 1 - mean similarity
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        top = np.take_along_axis(sim, idx, axis=1)
        out[s:e] = 1.0 - top.mean(axis=1)
    return out


def articulatory_vectors(artic_feats: Sequence[Dict[str, float]]) -> np.ndarray:
    """Codeword counts as a dense matrix, L2 normalised, for the second space."""
    keys = sorted({k for f in artic_feats for k in f})
    kidx = {k: i for i, k in enumerate(keys)}
    X = np.zeros((len(artic_feats), len(keys)), dtype=np.float32)
    for i, f in enumerate(artic_feats):
        for k, v in f.items():
            X[i, kidx[k]] = v
    return X


def summarize(name: str, d: np.ndarray, idx: Sequence[int], pool_mean: float) -> dict:
    v = d[list(idx)]
    return {
        "strategy": name,
        "n": len(idx),
        "mean_knn_dist": float(v.mean()),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "vs_pool": float(v.mean() - pool_mean),
        "frac_in_sparsest_decile": float((v >= np.percentile(d, 90)).mean()),
        "frac_in_densest_decile": float((v <= np.percentile(d, 10)).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--corpus", default="phoenix", choices=["phoenix", "isharah"])
    ap.add_argument("--isharah-split", default="SI")
    ap.add_argument("--budget", type=float, default=5.5)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strategies", nargs="+",
                    default=["random", "pc_art_only", "pc_rep_only", "pc_full",
                             "kcenter", "privileged_oracle_gloss"])
    a = ap.parse_args()

    from acquire import STRATEGIES
    from baselines_pc import PC_BASELINES
    from featurizer_pc import fit_pc_featurizer, pc_features
    from posecover import (ABLATIONS, generic_embeddings, quality_score,
                           select_posecover)

    root, poses = Path(a.root), Path(a.poses)
    if a.corpus == "isharah":
        from data_isharah import fill_durations, load_split
        train = load_split(poses, root, "train", a.isharah_split)
    else:
        from data_phoenix import fill_durations, load_split
        train = load_split(root, poses, "train")
    fill_durations(train)

    dur = np.array([c.duration_s for c in train])
    paths = [c.pose_path for c in train]
    srcs = [c.signer for c in train]
    print(f"corpus={a.corpus}  pool={len(train)}  budget={a.budget}h  k={a.k}",
          flush=True)

    print("building pool representations ...", flush=True)
    feat = fit_pc_featurizer(paths)
    artic = pc_features(feat, paths)
    emb = generic_embeddings(paths)
    qual = np.array([quality_score(np.load(p, allow_pickle=True)["ok"]) for p in paths])

    print("computing local density ...", flush=True)
    d_gen = knn_distance(emb, k=a.k)
    d_art = knn_distance(articulatory_vectors(artic), k=a.k)
    print(f"  generic:      mean {d_gen.mean():.4f}  sd {d_gen.std():.4f}")
    print(f"  articulatory: mean {d_art.mean():.4f}  sd {d_art.std():.4f}")

    bs = a.budget * 3600.0
    rows_gen, rows_art, sel_store = [], [], {}
    for s in a.strategies:
        if s.startswith("pc_"):
            idx = select_posecover(dur, bs, "translation", artic_feats=artic,
                                   embeddings=emb, quality=qual, sources=srcs,
                                   cfg=ABLATIONS[s[3:]])
        elif s in PC_BASELINES:
            idx = PC_BASELINES[s](dur, bs, "translation", embeddings=emb, sources=srcs,
                                  seed=0,
                                  translations=[" ".join(c.text) for c in train]
                                  if s == "privileged_text_fl" else None,
                                  glosses=[c.gloss for c in train]
                                  if s == "privileged_oracle_gloss" else None)
        else:
            idx = STRATEGIES[s](dur, bs, "translation", artic_feats=artic, seed=0)
        sel_store[s] = [int(i) for i in idx]
        rows_gen.append(summarize(s, d_gen, idx, d_gen.mean()))
        rows_art.append(summarize(s, d_art, idx, d_art.mean()))

    def show(title, rows, note):
        print(f"\n### {title}")
        print(note)
        print("%-26s %6s %10s %10s %10s %10s" %
              ("strategy", "n", "mean kNN", "vs pool", "%sparsest", "%densest"))
        for r in sorted(rows, key=lambda x: -x["mean_knn_dist"]):
            print("%-26s %6d %10.4f %+10.4f %9.1f%% %9.1f%%" %
                  (r["strategy"], r["n"], r["mean_knn_dist"], r["vs_pool"],
                   100 * r["frac_in_sparsest_decile"],
                   100 * r["frac_in_densest_decile"]))

    show("Generic pose embedding (informative)", rows_gen,
         "Larger mean kNN distance = sparser neighbourhood = more atypical.\n"
         "The comparison that carries weight is articulatory-only against random.\n"
         "Facility location scoring dense here is definitional, not a finding.")
    show("Articulatory codeword space (near-definitional)", rows_art,
         "Reported for completeness. The coverage objective up-weights rare codewords,\n"
         "so selecting clips that are rare in this space is close to tautological.")

    out = {"corpus": a.corpus, "budget_hours": a.budget, "k": a.k,
           "pool_size": len(train),
           "pool_mean_knn_generic": float(d_gen.mean()),
           "pool_mean_knn_articulatory": float(d_art.mean()),
           "generic": rows_gen, "articulatory": rows_art,
           "density_percentiles_generic": {
               str(p): float(np.percentile(d_gen, p)) for p in (10, 25, 50, 75, 90)},
           "selected": sel_store}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
