#!/bin/bash
#SBATCH --job-name=privslt
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS_GPU
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --output=logs/privslt_%j.out

# Cluster specific settings. Adjust the SBATCH lines for your scheduler, then set:
#   PROJECT  directory holding src/, data/, poses/ and runs/
#   ENV      conda environment prefix created by mkenv.sh
: "${PROJECT:?set PROJECT to the project root}"
: "${ENV:?set ENV to the conda environment prefix}"

set -uo pipefail
module load conda   # site specific

# The four empty privileged cells in main-text Table 1. One seed each, matching how the
# privileged rows were run for recognition. Settings copied verbatim from the jobs that
# produced the rest of each translation column: legacy descriptors at 400 handshape
# codewords for PHOENIX, spec descriptors for Isharah.

echo "########## 1. PHOENIX14T translation, privileged ##########"
conda run --no-capture-output -p $ENV python $PROJECT/src/run_budget.py \
  --root $PROJECT/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T --poses $PROJECT/poses \
  --annotation translation --out $PROJECT/runs/pcfull --archs gloss_free \
  --budgets-hours 0.5 1 2 5 --seeds 0 \
  --n-handshape-clusters 400 --max-epochs 60 \
  --strategies privileged_text_fl privileged_oracle_gloss

echo "########## 2. Isharah translation, privileged ##########"
conda run --no-capture-output -p $ENV python $PROJECT/src/run_budget.py \
  --corpus isharah --isharah-split SI \
  --root $PROJECT/data/isharah/annotations --poses $PROJECT/poses_isharah \
  --descriptors spec --annotation translation --out $PROJECT/runs/isharah_bleu \
  --archs gloss_free --budgets-hours 7.1 14.2 28.4 --seeds 0 --max-epochs 60 \
  --strategies privileged_text_fl privileged_oracle_gloss
echo "ALL DONE"
