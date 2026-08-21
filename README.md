# Coverage Is Not Enough

Code for the paper "Coverage Is Not Enough: A Controlled Study of Annotation
Selection for Low-Resource Sign Language".

The study asks a practical question. A corpus team has a fixed annotation budget
and a large pool of unlabelled sign language video. Which clips should they pay to
annotate? A natural answer is to buy clips that add previously uncovered
articulatory patterns. This code tests that answer against random sampling and
finds it does not hold.

Everything here is label free at selection time. A static analysis test enforces
that: `tests/test_pc.py` parses every selection module and fails if it reads a
gloss or translation field.

## Contents

```
src/          library and entry points
scripts/      SLURM job scripts, one per experiment group
tests/        test suite, including the label isolation check
requirements.txt
```

## 1. Environment

Python 3.11. The environment is created by `scripts/mkenv.sh`, which you can run
directly rather than through the scheduler.

```bash
export ENV=$HOME/.conda/envs/posecover
bash scripts/mkenv.sh
```

Or by hand:

```bash
conda create -y -p "$ENV" python=3.11
conda run -p "$ENV" pip install torch --index-url https://download.pytorch.org/whl/cu128
conda run -p "$ENV" pip install numpy scipy scikit-learn mediapipe \
    opencv-python-headless matplotlib tqdm sacrebleu
```

Install torch first. If numpy is installed first, pip may pin a version that the
torch wheel then downgrades, which breaks scipy and scikit-learn at import time.

Verify:

```bash
conda run -p "$ENV" python -c "
import torch, mediapipe
from mediapipe.tasks.python import vision
print(torch.__version__, torch.cuda.is_available())
print(hasattr(vision, 'HolisticLandmarker'))
"
```

Both lines must print True. Exact versions used are in `requirements.txt`.

### Known version constraints

**MediaPipe 1.0 removed `mp.solutions`.** Pose extraction uses the Tasks API
(`mediapipe.tasks.python.vision.HolisticLandmarker`). Code written against the
older `mp.solutions.holistic` interface will not run.

**Timestamps must increase monotonically across a whole landmarker session.** If
you reuse one landmarker across clips, restarting timestamps at zero for each clip
raises an error partway through a batch. `extract_pose.py` threads a global offset
for this reason.

## 2. Data

Two corpora, obtained separately under their own licences.

**PHOENIX14T.** German Sign Language weather broadcasts. 7,096 training clips,
9.19 hours of video. Point `--root` at the directory containing `annotations/` and
`features/`.

**Isharah.** Saudi Sign Language, recorded on smartphones in unconstrained
settings. This study uses the signer independent split: a 10,000 clip training
pool, 23.64 hours, 13 signers. Isharah ships pose as pickled arrays rather than
video, so it needs one conversion pass.

Expected layout:

```
$PROJECT/
  data/
    PHOENIX-2014-T-release-v3/PHOENIX-2014-T/
    isharah/annotations/
  poses/            PHOENIX14T pose npz, one per clip
  poses_isharah/    Isharah pose npz, one per clip
  runs/             results, one json per experiment cell
  logs/
```

## 3. Pose extraction

PHOENIX14T, from video. This is the slow step. It shards, so run it as an array
job:

```bash
python src/extract_pose.py --root $PROJECT/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --out $PROJECT/poses --split train --shard $SLURM_ARRAY_TASK_ID --n-shards 32
```

Isharah, from the distributed pickle:

```bash
python src/isharah_to_npz.py --pkl path/to/isharah_poses.pkl \
    --annotations $PROJECT/data/isharah/annotations --out $PROJECT/poses_isharah
```

Each output npz holds landmark arrays plus an `ok` mask marking frames where a
given body part was detected. The mask matters: frames with no detected hand are
excluded before quantisation. Including them creates a codeword that means "hand
not found", which a coverage objective then treats as a rare pattern worth buying.

Check detection quality before going further:

```bash
python src/det_by_signer.py $PROJECT/poses/train
```

Detection rates vary by signer. That variation is reported in the paper as a
fairness concern, since a noise sensitive selector can allocate annotation partly
according to measurement quality.

## 4. Running the experiments

`run_budget.py` is the main driver. One invocation covers a grid of budgets,
strategies and seeds, writing one json per cell. Cells are resumable: an existing
output file is skipped, so a preempted job continues rather than restarting.

```bash
python src/run_budget.py \
    --root $PROJECT/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --poses $PROJECT/poses \
    --out $PROJECT/runs/spec_wer \
    --task gloss_ctc --descriptors spec --annotation translation \
    --budgets-hours 2.8 5.5 11.0 --seeds 0 1 2 --max-epochs 60 \
    --strategies random source_random kcenter \
                 pc_rep_only pc_art_only pc_art_rep pc_full
```

Key arguments:

| Argument | Meaning |
|---|---|
| `--task` | `gloss_ctc` scores WER, `translation` scores BLEU |
| `--corpus` | `phoenix` or `isharah` |
| `--descriptors` | `spec` for the six group featurizer, `legacy` for the earlier five group one |
| `--annotation` | cost model used to price the budget, `translation` at 6x realtime or `gloss` at 40x |
| `--budgets-hours` | annotator hours, not video hours |
| `--select-only` | run selection and coverage diagnostics without training |

Budgets are annotator hours under the chosen cost model. The paper uses the same
fraction of pool cost on both corpora rather than the same absolute hours, because
the Isharah pool is 2.6 times larger.

### Strategies

Label free:

```
random               uniform until the budget is spent
source_random        budget split across recording sources, then uniform within
kcenter              greedy k-center on a generic pose embedding
pc_rep_only          facility location, favours clips similar to many others
pc_art_only          articulatory coverage alone
pc_art_rep           articulatory coverage plus representativeness
pc_full              0.60 articulatory + 0.30 representativeness + 0.10 source
pc_full_no_rare      full objective without the rarity weight
pc_full_no_qual      full objective without the pose quality weight
dens10_art_only      articulatory coverage on a pool with the sparsest 10 percent removed
dens20_art_only      the same at 20 percent
dens10_full          full objective, sparsest 10 percent removed
dens20_full          the same at 20 percent
```

Model dependent, run through their own drivers because they need a seed model:

```bash
python src/run_entropy.py --root ... --poses ... --out ... --budgets-hours 5.5 11.0 --seeds 0 1 2
python src/run_badge.py   --root ... --poses ... --out ... --budgets-hours 2.8 5.5 11.0 --seeds 0 1 2
```

Privileged diagnostics. These read hidden labels and are not label free. They are
upper bounds for interpreting the label free results, not competitors:

```
privileged_text_fl          facility location on hidden translations
privileged_oracle_gloss     favours gloss types not yet covered
```

`run_budget.py` prints a warning when a privileged selector is used.

## 5. Why does coverage fail

Two parts, reported in the paper section of the same name.

**Part one, local density.** For every pool clip, mean cosine distance to its 20
nearest neighbours. Larger distance means a sparser neighbourhood, so a more
atypical clip.

```bash
python src/density.py --corpus phoenix \
    --root $PROJECT/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --poses $PROJECT/poses --budget 5.5 --k 20 \
    --out $PROJECT/runs/density_phoenix.json
```

The output json includes `selected`, the actual purchased clip indices per
strategy. Every statistic in the paper is derived from those, so any of them can
be recomputed without rerunning selection.

**Duration control.** Informed selectors buy more, shorter clips at equal cost. If
clip length and density were related, the density result would restate a known
effect rather than add one. This recomputes the comparison within duration
quartiles:

```bash
python src/dur_conf.py --corpus phoenix --root ... --poses ... \
    --density-json $PROJECT/runs/density_phoenix.json --k 20
```

**Part two, density filtered selection.** Removes the sparsest candidates and
reruns the identical selector on what remains. Objective, weights, greedy
procedure and budget are unchanged; only the candidate set differs.

```bash
python src/run_budget.py --corpus phoenix --task gloss_ctc --descriptors spec \
    --root ... --poses ... --out $PROJECT/runs/spec_wer \
    --annotation translation --budgets-hours 2.8 5.5 11.0 --seeds 0 1 2 \
    --strategies dens10_art_only dens20_art_only dens10_full dens20_full
```

Note when interpreting a positive result here: filtering by density in the generic
embedding is itself a representativeness constraint, applied as a hard pre filter
rather than as a term in the objective. An improvement means coverage needs a
representativeness constraint to be usable, not that coverage works once the data
is cleaned.

## 6. Supporting diagnostics

**Are the articulatory channels meaningful, or do they encode signer identity?**
Clustering handshape descriptors can converge on hand size or camera distance.
Such clusters still produce a coverage curve, so the failure is invisible
downstream. This measures excess normalised mutual information between cluster
assignment and signer against a permutation null, with thresholds calibrated on
synthetic pose where the answer is known by construction:

```bash
python src/cluster_gate.py --poses $PROJECT/poses/train --n-clips 1200 --json-out gate.json
```

**How stable is each selection under pose noise?** Selects from clean and
corrupted pose and reports duration weighted overlap:

```bash
python src/corruption_study.py --root ... --poses ... --out $PROJECT/runs/corrupt.json \
    --budget-hours 5.5 --strategies pc_art_only pc_full pc_rep_only
```

**Does detection quality correlate with per signer performance?**

```bash
python src/fairness_analysis.py --poses $PROJECT/poses --runs $PROJECT/runs --json-out fairness.json
```

## 7. Tables

```bash
python src/paper_tables.py --runs $PROJECT/runs/spec_wer \
    --root ... --poses ... --budgets 2.8 5.5 11.0 --out tables.tex
```

This re-runs selection to measure properties of the purchased subsets, because the
selected clip ids are not stored in the training output. It is CPU only and takes
roughly forty minutes per corpus.

## 8. Tests

```bash
python -m pytest tests/test_pc.py -v
```

The important one is the label isolation test. It parses each selection module and
fails if a gloss or translation field is read, by attribute or by subscript. Naming
an annotation type as a dictionary key is allowed, since the cost model has to know
the relative price of a gloss and a translation.

## Reproducing the paper

| Result | Script |
|---|---|
| Main grids, PHOENIX14T | `scripts/full.sh` |
| Main grids, Isharah | `scripts/isharah_full.sh` |
| Density analysis, part one | `scripts/dens.sh` |
| Duration control | `scripts/dur_conf.sh` |
| Density filtered selection, part two | `scripts/densgrid.sh` |
| Articulatory channel validation | `scripts/gate.sh`, `scripts/gate_ish.sh` |
| Corruption stability | `scripts/corrupt.sh` |
| Coverage saturation sweep | `scripts/satsweep.sh` |
| Remaining Table 1 cells | `scripts/table1gaps.sh`, `scripts/privslt.sh` |
| Tests | `scripts/runtests.sh` |

Each script is written for SLURM. Set `PROJECT` and `ENV`, and adjust the SBATCH
account, QOS and partition lines for your site. The scripts refuse to run if those
two variables are unset.

Compute used: one GPU at a time. Recognition cells take one to four minutes each,
translation cells five to thirty. The GPU is not the bottleneck; utilisation sits
near ten percent and the limit is dataloader throughput, so a modest GPU with more
CPU workers will outperform a large GPU with few.

## Licence and data

The code is released for research use. PHOENIX14T and Isharah are distributed by
their respective authors under their own terms and are not included here.
