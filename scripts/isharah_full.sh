#!/bin/bash
#SBATCH --job-name=ish_full
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS_GPU
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --dependency=afterany:38894066
#SBATCH --output=logs/ishfull_%j.out

# Cluster specific settings. Adjust the SBATCH lines for your scheduler, then set:
#   PROJECT  directory holding src/, data/, poses/ and runs/
#   ENV      conda environment prefix created by mkenv.sh
: "${PROJECT:?set PROJECT to the project root}"
: "${ENV:?set ENV to the conda environment prefix}"

set -uo pipefail
module load conda   # site specific
ANN=$PROJECT/data/isharah/annotations
# Isharah SI train is 23.64h; full annotation costs 141.8 annotator-hours, so
# 5/10/20% is 7.1/14.2/28.4 -- the same FRACTION as on PHOENIX14T.
B="7.1 14.2 28.4"

ish () { conda run --no-capture-output -p $ENV python $PROJECT/src/run_budget.py \
           --corpus isharah --isharah-split SI --root $ANN --poses $PROJECT/poses_isharah \
           --descriptors spec --annotation translation --budgets-hours $B "$@" ; }

echo "########## A. Isharah WER, every label-free strategy ##########"
ish --out $PROJECT/runs/isharah_wer --task gloss_ctc --seeds 0 1 2 --max-epochs 60 \
    --strategies random source_random kcenter \
                 pc_rep_only pc_art_only pc_art_rep pc_full

echo "########## B. Isharah WER, privileged upper bounds ##########"
ish --out $PROJECT/runs/isharah_wer --task gloss_ctc --seeds 0 --max-epochs 60 \
    --strategies privileged_text_fl privileged_oracle_gloss

echo "########## C. Isharah WER, entropy AL ##########"
conda run --no-capture-output -p $ENV python $PROJECT/src/run_entropy.py \
  --root $ANN --poses $PROJECT/poses_isharah --out $PROJECT/runs/isharah_wer \
  --budgets-hours 14.2 28.4 --seeds 0 1 2 --max-epochs 60 --corpus isharah

echo "########## D. Isharah WER, BADGE ##########"
conda run --no-capture-output -p $ENV python $PROJECT/src/run_badge.py \
  --root $ANN --poses $PROJECT/poses_isharah --out $PROJECT/runs/isharah_wer \
  --budgets-hours 7.1 14.2 28.4 --seeds 0 1 2 --max-epochs 60 --corpus isharah

echo "########## E. Isharah translation/BLEU ##########"
ish --out $PROJECT/runs/isharah_bleu --archs gloss_free --seeds 0 1 2 --max-epochs 60 \
    --strategies random kcenter pc_rep_only pc_art_only pc_full
echo "ALL DONE"
