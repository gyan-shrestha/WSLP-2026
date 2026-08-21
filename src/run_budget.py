"""Budget grid driver: select under budget -> train -> evaluate -> record.

Every run is one (budget, strategy, annotation type, architecture, seed) cell. Cells
are written to disk as they complete so a preempted job resumes instead of restarting,
and so partial grids are still plottable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from acquire import (COST_REALTIME, STRATEGIES, clip_cost, coverage_profile,
                     fit_featurizer, artic_features)
from data_phoenix import build_vocabs, fill_durations, load_split
from models import ModelConfig
from train import TrainConfig, train_one
from data_phoenix import FEATURE_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets-hours", type=float, nargs="+",
                    default=[2, 5, 10, 20, 40])
    ap.add_argument("--strategies", nargs="+",
                    default=["random", "duration_random", "phon_coverage"])
    ap.add_argument("--annotation", default="translation", choices=list(COST_REALTIME))
    ap.add_argument("--archs", nargs="+", default=["gloss_free"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--max-epochs", type=int, default=120)
    ap.add_argument("--pool-limit", type=int, default=0)
    ap.add_argument("--n-handshape-clusters", type=int, default=40,
                    help="inventory granularity. At 40, random selection already "
                         "covers ~97%% of the inventory at moderate budgets, leaving "
                         "the coverage objective nothing to win, see the saturation "
                         "analysis. Finer codebooks keep coverage scarce.")
    ap.add_argument("--select-only", action="store_true",
                    help="run selection and coverage diagnostics without training")
    ap.add_argument("--task", default="translation", choices=["translation", "gloss_ctc"],
                    help="translation scores BLEU/chrF with a pool-wide vocabulary; "
                         "gloss_ctc scores WER under the specified protocol, where the "
                         "vocabulary comes only from the purchased subset")
    ap.add_argument("--corpus", default="phoenix", choices=["phoenix", "isharah"],
                    help="phoenix: PHOENIX14T (DGS, weather, 9.19h). isharah: Isharah "
                         "(Saudi SL, unconstrained smartphone capture, 23.64h, 13 "
                         "signers, signer-independent split). Running both is what "
                         "separates a property of the selection problem from a "
                         "property of repetitive weather broadcasts.")
    ap.add_argument("--isharah-split", default="SI", choices=["SI", "US"],
                    help="SI: signer-independent. US: unseen sentences.")
    ap.add_argument("--descriptors", default="legacy", choices=["legacy", "spec"],
                    help="legacy: the 5-group featurizer with a configurable handshape "
                         "codebook. spec: the 6 groups of paper Eq. 7 at the codebook "
                         "sizes of impl. note 6 (64/16/24/32/8/16), with the shoulder "
                         "rotation of Eq. 4 and the stable-frame rule of 6.1.")
    args = ap.parse_args()

    root, poses = Path(args.root), Path(args.poses)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    if args.corpus == "isharah":
        from data_isharah import fill_durations as ish_fill
        from data_isharah import load_split as ish_load
        # --root points at the annotations directory for this corpus
        train = ish_load(poses, root, "train", args.isharah_split)
        dev = ish_load(poses, root, "dev", args.isharah_split)
        if args.pool_limit:
            train = train[: args.pool_limit]
        ish_fill(train); ish_fill(dev)
    else:
        train = load_split(root, poses, "train")
        dev = load_split(root, poses, "dev")
        if args.pool_limit:
            train = train[: args.pool_limit]
        fill_durations(train); fill_durations(dev)
    gv, tv = build_vocabs(train)

    dur = np.array([c.duration_s for c in train])
    print(f"pool={len(train)} clips  {dur.sum()/3600:.2f}h video  "
          f"dev={len(dev)}  gloss_vocab={len(gv)}  text_vocab={len(tv)}", flush=True)
    print(f"full-pool cost at '{args.annotation}': "
          f"{dur.sum()*COST_REALTIME[args.annotation]/3600:.1f} annotator-hours", flush=True)

    paths = [c.pose_path for c in train]

    need_artic = any(s.startswith("phon_coverage") or s.startswith("pc_")
                     or s.startswith("dens") for s in args.strategies)
    artic = None
    if need_artic:
        t0 = time.time()
        if args.descriptors == "spec":
            from featurizer_pc import fit_pc_featurizer, pc_features
            feat = fit_pc_featurizer(paths)
            artic = pc_features(feat, paths)
        else:
            feat = fit_featurizer(paths, n_hs=args.n_handshape_clusters)
            artic = artic_features(feat, paths)
        n_tok = len(set(k for f in artic for k in f))
        print(f"articulatory features ({args.descriptors}) ready in "
              f"{(time.time()-t0)/60:.1f} min: {n_tok} distinct tokens", flush=True)

    # The generic embedding, quality scores and source ids are pool properties: they
    # do not depend on budget, strategy or seed, so they are built once and shared.
    # The extra baselines need them too, not only the PoseCover ablations.
    from baselines_pc import PC_BASELINES, PRIVILEGED
    from posecover import ABLATIONS, generic_embeddings, quality_score, select_posecover

    needs_pool = [s for s in args.strategies
                  if s.startswith("pc_") or s.startswith("dens")
                  or s in PC_BASELINES]
    emb = qual = srcs = None
    if needs_pool:
        t0 = time.time()
        emb = generic_embeddings(paths)
        qual = np.array([quality_score(np.load(p, allow_pickle=True)["ok"]) for p in paths])
        srcs = [c.signer for c in train]
        print(f"pool inputs ready in {(time.time()-t0)/60:.1f} min: "
              f"emb={emb.shape} quality mean={qual.mean():.3f} "
              f"min={qual.min():.3f} sources={len(set(srcs))}", flush=True)

    used_privileged = sorted(set(args.strategies) & PRIVILEGED)
    if used_privileged:
        print(f"WARNING privileged selectors in this run: {', '.join(used_privileged)}. "
              f"These read hidden labels and are diagnostic upper bounds, NOT "
              f"label-free competitors. Report them separately.", flush=True)

    dens_masks: dict = {}   # drop fraction -> keep mask, cached per pool
    cache: dict = {}
    for seed in args.seeds:
        for budget_h in args.budgets_hours:
            budget_s = budget_h * 3600.0
            for strat in args.strategies:
                if strat.startswith("dens"):
                    # Density-filtered variants: identical objective and greedy, run on
                    # a pool with the sparsest candidates removed. The mask is cached
                    # per drop fraction since it depends only on the pool.
                    from density_filter import parse_strategy, density_mask, \
                        select_density_filtered
                    parsed = parse_strategy(strat)
                    if parsed is None:
                        print(f"  [skip] unknown density strategy {strat}", flush=True)
                        continue
                    drop, dcfg = parsed
                    if drop not in dens_masks:
                        dens_masks[drop] = density_mask(emb, drop_frac=drop)
                        print(f"  density mask drop={drop:.0%}: "
                              f"{dens_masks[drop].sum()}/{len(emb)} candidates kept",
                              flush=True)
                    idx = select_density_filtered(
                        dur, budget_s, args.annotation, artic_feats=artic,
                        embeddings=emb, quality=qual, sources=srcs, cfg=dcfg,
                        precomputed_mask=dens_masks[drop])
                elif strat.startswith("pc_"):
                    idx = select_posecover(dur, budget_s, args.annotation,
                                           artic_feats=artic, embeddings=emb,
                                           quality=qual, sources=srcs,
                                           cfg=ABLATIONS[strat[3:]])
                elif strat in PC_BASELINES:
                    # privileged selectors get the hidden labels here and nowhere else;
                    # they are diagnostics, and the warning above marks them in the log
                    idx = PC_BASELINES[strat](
                        dur, budget_s, args.annotation, embeddings=emb, sources=srcs,
                        seed=seed,
                        translations=[" ".join(c.text) for c in train]
                        if strat == "privileged_text_fl" else None,
                        glosses=[c.gloss for c in train]
                        if strat == "privileged_oracle_gloss" else None)
                else:
                    idx = STRATEGIES[strat](dur, budget_s, args.annotation,
                                            artic_feats=artic, seed=seed)
                if not idx:
                    print(f"  [skip] {strat} b={budget_h}h selected nothing", flush=True)
                    continue
                spent = sum(clip_cost(dur[i], args.annotation) for i in idx)
                purchase = {train[i].clip_id: args.annotation for i in idx}
                prof = coverage_profile(artic, idx) if artic else {}

                if args.select_only:
                    rec = dict(prof, strategy=strat, budget_hours=budget_h, seed=seed,
                               n_clips=len(idx), spent_hours=spent / 3600.0,
                               mean_duration=float(dur[idx].mean()),
                               n_signers=len({train[i].signer for i in idx}))
                    (out_dir / f"sel_{strat}_b{budget_h}_s{seed}.json").write_text(
                        json.dumps(rec, indent=2))
                    print(f"  {strat:16s} b={budget_h:5.1f}h n={len(idx):5d} "
                          f"meandur={rec['mean_duration']:5.2f}s "
                          f"cov_all={prof.get('cov_all', float('nan')):.4f} "
                          f"cov_HS={prof.get('cov_HS', float('nan')):.4f}", flush=True)
                    continue

                if args.task == "gloss_ctc":
                    from train_ctc import CTCTrainConfig, train_ctc_one
                    tag = f"ctc_{args.annotation}_{strat}_b{budget_h}_s{seed}"
                    fp = out_dir / f"{tag}.json"
                    if fp.exists():
                        print(f"  [have] {tag}", flush=True)
                        continue
                    print(f"\n=== {tag}: {len(idx)} clips, "
                          f"{spent/3600:.2f}/{budget_h}h used ===", flush=True)
                    res = train_ctc_one(
                        purchase, train, dev,
                        cfg=CTCTrainConfig(seed=seed, max_epochs=args.max_epochs),
                        cache=cache, log_prefix=f"  [{tag}] ")
                    res.update(strategy=strat, annotation=args.annotation,
                               budget_hours=budget_h, spent_hours=spent / 3600.0,
                               seed=seed, **prof)
                    fp.write_text(json.dumps(res, indent=2))
                    print(f"  -> WER {res['dev']['wer']:.2f}  "
                          f"(del {res['dev']['del']:.1f}) vocab {res['gloss_vocab_size']} "
                          f"OOV {res['dev_oov_token_pct']:.1f}%  "
                          f"({res['minutes']:.1f} min)", flush=True)
                    continue

                for arch in args.archs:
                    tag = f"{arch}_{args.annotation}_{strat}_b{budget_h}_s{seed}"
                    fp = out_dir / f"{tag}.json"
                    if fp.exists():
                        print(f"  [have] {tag}", flush=True)
                        continue
                    print(f"\n=== {tag}: {len(idx)} clips, "
                          f"{spent/3600:.2f}/{budget_h}h used ===", flush=True)
                    res = train_one(
                        purchase, train, dev, gv, tv, arch,
                        tcfg=TrainConfig(seed=seed, max_epochs=args.max_epochs),
                        mcfg=ModelConfig(feature_dim=FEATURE_DIM),
                        cache=cache, log_prefix=f"  [{tag}] ")
                    res.update(strategy=strat, annotation=args.annotation,
                               budget_hours=budget_h, spent_hours=spent / 3600.0,
                               seed=seed, **prof)
                    fp.write_text(json.dumps(res, indent=2))
                    print(f"  -> BLEU {res['dev']['bleu']:.2f}  chrF {res['dev']['chrf']:.2f}  "
                          f"cov_all {prof.get('cov_all', float('nan')):.3f}  "
                          f"({res['minutes']:.1f} min)", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
