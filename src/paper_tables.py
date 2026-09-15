"""Produce Tables 1 to 3 of the PoseCover paper, in its format, from our runs.

Table 1 needs WER at three budgets plus AULC, OOV and rare-gloss recall.
Table 2 needs properties of the selected set itself: gloss types acquired, rare and
singleton types, source entropy, pose quality, redundancy.
Table 3 is the objective ablation.

WER and OOV come from the saved training runs. Everything in Table 2 is a property of
the purchase rather than of the model, and the selected clip ids were not stored, so
selection is re-run here. That is cheap and exact: every selector except `random` is
deterministic, and `random` is seeded, so the reproduced subsets are the same ones the
models were trained on.

Rare-gloss recall needs model hypotheses, which were not saved either. It is reported
as unavailable rather than approximated, since a guessed number in a results table is
worse than an honest gap.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from acquire import clip_cost
from metrics_pc import aulc, redundancy, selected_type_frequency, source_entropy, vocab_metrics

# display name -> internal strategy key, in the order the paper lists them
ROWS = [
    ("Random", "random", "Duration"),
    ("Source-random", "source_random", "Duration + source"),
    ("Pose $k$-center", "kcenter", "Unlabeled pose"),
    ("Pose facility location", "pc_rep_only", "Unlabeled pose"),
    ("Articulatory coverage", "pc_art_only", "Unlabeled pose"),
    ("\\textsc{PoseCover}", "pc_full", "Unlabeled pose + source"),
    ("Entropy active learning", "entropy_al", "2\\% labeled seed"),
    ("BADGE", "badge", "2\\% labeled seed"),
    ("Text facility location", "privileged_text_fl", "Privileged translation"),
    ("Oracle gloss coverage", "privileged_oracle_gloss", "Privileged glosses"),
]

ABLATIONS = [
    ("Representativeness only", "pc_rep_only"),
    ("Articulatory coverage only", "pc_art_only"),
    ("Articulatory + representativeness", "pc_art_rep"),
    ("Full without rare weighting", "pc_full_no_rare"),
    ("Full without quality weighting", "pc_full_no_qual"),
    ("Full \\textsc{PoseCover}", "pc_full"),
]


def load_runs(run_dir: Path):
    out = defaultdict(list)
    for f in glob.glob(f"{run_dir}/*.json"):
        d = json.load(open(f))
        if "dev" not in d or "wer" not in d["dev"]:
            continue
        out[(d["strategy"], d["budget_hours"])].append(d)
    return out


def ms(vals):
    if not vals:
        return None, None
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def fmt(m, s, dec=2):
    return "--" if m is None else (f"{m:.{dec}f}" if s in (None, 0.0)
                                   else f"{m:.{dec}f}\\,\\tiny{{$\\pm${s:.1f}}}")


def selection_properties(strategies, budgets, corpus, root, poses, isharah_split="SI"):
    """Re-run selection and measure properties of the purchased set."""
    from posecover import (ABLATIONS as ABL_CFG, generic_embeddings, quality_score,
                           select_posecover)
    from acquire import STRATEGIES
    from baselines_pc import PC_BASELINES
    from featurizer_pc import fit_pc_featurizer, pc_features

    if corpus == "isharah":
        from data_isharah import fill_durations, load_split
        train = load_split(poses, root, "train", isharah_split)
    else:
        from data_phoenix import fill_durations, load_split
        train = load_split(root, poses, "train")
    fill_durations(train)

    dur = np.array([c.duration_s for c in train])
    paths = [c.pose_path for c in train]
    srcs = [c.signer for c in train]
    pool_gloss = [c.gloss for c in train]

    print("computing pool inputs ...", flush=True)
    feat = fit_pc_featurizer(paths)
    artic = pc_features(feat, paths)
    emb = generic_embeddings(paths)
    qual = np.array([quality_score(np.load(p, allow_pickle=True)["ok"]) for p in paths])

    out = {}
    for strat in strategies:
        for b in budgets:
            bs = b * 3600.0
            try:
                if strat.startswith("pc_"):
                    idx = select_posecover(dur, bs, "translation", artic_feats=artic,
                                           embeddings=emb, quality=qual, sources=srcs,
                                           cfg=ABL_CFG[strat[3:]])
                elif strat in PC_BASELINES:
                    idx = PC_BASELINES[strat](
                        dur, bs, "translation", embeddings=emb, sources=srcs, seed=0,
                        translations=[" ".join(c.text) for c in train]
                        if strat == "privileged_text_fl" else None,
                        glosses=pool_gloss if strat == "privileged_oracle_gloss" else None)
                elif strat in STRATEGIES:
                    idx = STRATEGIES[strat](dur, bs, "translation",
                                            artic_feats=artic, seed=0)
                else:
                    continue
            except Exception as e:
                print(f"  skip {strat} @ {b}: {e}")
                continue

            sel_gloss = [train[i].gloss for i in idx]
            vm = vocab_metrics(sel_gloss, pool_gloss, pool_gloss)
            freq = selected_type_frequency(sel_gloss, min_count=5)
            out[(strat, b)] = {
                "n_clips": len(idx),
                "gloss_types": vm["n_selected_types"],
                "gloss_coverage": vm["gloss_coverage"],
                "rare_coverage": vm["rare_coverage"],
                "singleton_coverage": vm["singleton_coverage"],
                "source_entropy": source_entropy([train[i].signer for i in idx]),
                "pose_quality": float(qual[idx].mean()),
                "redundancy": redundancy(emb, idx),
                "tokens_per_type": freq["tokens_per_type"],
                "types_ge5": freq["n_types_ge_min"],
                "frac_types_ge5": freq["frac_types_ge_min"],
            }
            print(f"  {strat:24s} b={b:5.1f}  n={len(idx):5d}  "
                  f"types={vm['n_selected_types']:4d}  "
                  f"H_src={out[(strat,b)]['source_entropy']:.3f}  "
                  f"redund={out[(strat,b)]['redundancy']:.3f}  "
                  f"tok/type={freq['tokens_per_type']:.2f}  "
                  f"types>=5={freq['n_types_ge_min']:4d}", flush=True)
    return out


def table1(runs, budgets, props, mid):
    L = []
    L.append("\\begin{table*}[t]\\centering\\small")
    L.append("\\begin{tabular}{llrrrrrr}")
    L.append("\\toprule")
    L.append("Method & Selection information & WER@5\\% $\\downarrow$ & WER@10\\% $\\downarrow$ "
             "& WER@20\\% $\\downarrow$ & AULC $\\uparrow$ & OOV@10\\% $\\downarrow$ "
             "& Types@10\\% $\\uparrow$ \\\\")
    L.append("\\midrule")
    for name, key, info in ROWS:
        if key == "privileged_text_fl":
            L.append("\\midrule")
        cells, wers = [], []
        for b in budgets:
            m, s = ms([d["dev"]["wer"] for d in runs.get((key, b), [])])
            cells.append(fmt(m, s))
            wers.append(m)
        a = (f"{aulc([b for b, w in zip(budgets, wers) if w is not None], [w for w in wers if w is not None]):.3f}"
             if sum(w is not None for w in wers) >= 2 else "--")
        om, os_ = ms([d.get("dev_oov_token_pct", float('nan'))
                      for d in runs.get((key, mid), [])])
        ty = props.get((key, mid), {}).get("gloss_types")
        L.append(f"{name} & {info} & " + " & ".join(cells) +
                 f" & {a} & {'--' if om is None else f'{om:.2f}'} "
                 f"& {'--' if ty is None else ty} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\caption{Primary annotation-efficiency results on PHOENIX14T, three seeds. "
             "Budgets are 5, 10 and 20 percent of pool annotation cost. Privileged methods "
             "below the rule read hidden labels and are diagnostic upper bounds rather than "
             "label-free competitors.}")
    L.append("\\label{tab:main}\\end{table*}")
    return "\n".join(L)


def table2(props, mid):
    L = ["\\begin{table*}[t]\\centering\\small", "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
         "Method at 10\\% & Clips & Gloss types $\\uparrow$ & Gloss cov. $\\uparrow$ & "
         "Rare cov. $\\uparrow$ & Source entropy $\\uparrow$ & Pose quality $\\uparrow$ & "
         "Redundancy $\\downarrow$ & Tokens/type $\\uparrow$ & Types $\\geq$5 $\\uparrow$ \\\\",
         "\\midrule"]
    for name, key, _ in ROWS:
        p = props.get((key, mid))
        if not p:
            continue
        if key == "privileged_text_fl":
            L.append("\\midrule")
        L.append(f"{name} & {p['n_clips']} & {p['gloss_types']} & "
                 f"{p['gloss_coverage']:.3f} & {p['rare_coverage']:.3f} & "
                 f"{p['source_entropy']:.3f} & {p['pose_quality']:.3f} & "
                 f"{p['redundancy']:.3f} & {p['tokens_per_type']:.2f} & "
                 f"{p['types_ge5']} \\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          "\\caption{Post-selection analysis of the purchased set at the 10 percent budget. "
          "Gloss and rare-type coverage are computed after selection and were not available "
          "to any label-free selector. Tokens/type and Types $\\geq$5 describe repetition "
          "within the purchase itself: how much support each acquired gloss type gets, "
          "rather than how many distinct types were acquired.}",
          "\\label{tab:selection}\\end{table*}"]
    return "\n".join(L)


def table3(runs, budgets, props, mid):
    L = ["\\begin{table}[t]\\centering\\small", "\\begin{tabular}{lrrr}", "\\toprule",
         "Configuration at 10\\% & WER $\\downarrow$ & OOV $\\downarrow$ & "
         "Gloss cov. $\\uparrow$ \\\\", "\\midrule"]
    for name, key in ABLATIONS:
        m, s = ms([d["dev"]["wer"] for d in runs.get((key, mid), [])])
        om, _ = ms([d.get("dev_oov_token_pct", float('nan'))
                    for d in runs.get((key, mid), [])])
        gc = props.get((key, mid), {}).get("gloss_coverage")
        L.append(f"{name} & {fmt(m, s)} & {'--' if om is None else f'{om:.2f}'} & "
                 f"{'--' if gc is None else f'{gc:.3f}'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          "\\caption{Objective ablations at the 10 percent budget on PHOENIX14T.}",
          "\\label{tab:ablation}\\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--corpus", default="phoenix")
    ap.add_argument("--budgets", type=float, nargs="+", default=[2.8, 5.5, 11.0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    runs = load_runs(Path(a.runs))
    mid = a.budgets[1]
    keys = sorted({k for k, _ in runs})
    props = selection_properties(keys, [mid], a.corpus, Path(a.root), Path(a.poses))

    txt = "\n\n".join([table1(runs, a.budgets, props, mid),
                       table2(props, mid),
                       table3(runs, a.budgets, props, mid)])
    Path(a.out).write_text(txt)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
