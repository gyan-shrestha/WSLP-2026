#!/bin/bash
#SBATCH --job-name=pc_corrupt
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS_CPU
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --output=logs/corrupt_%j.out

# Cluster specific settings. Adjust the SBATCH lines for your scheduler, then set:
#   PROJECT  directory holding src/, data/, poses/ and runs/
#   ENV      conda environment prefix created by mkenv.sh
: "${PROJECT:?set PROJECT to the project root}"
: "${ENV:?set ENV to the conda environment prefix}"

set -uo pipefail
module load conda   # site specific
# Selection only, so this runs on the burst queue and does not touch the GPU that the
# main matrix is using.
conda run --no-capture-output -p $ENV \
  python $PROJECT/src/corruption_study.py \
    --root $PROJECT/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --poses $PROJECT/poses --cache-root $PROJECT/poses_corrupt \
    --out $PROJECT/runs/corruption_study.json \
    --budget-hours 5.5 --pool-limit 2000 \
    --strategies pc_full pc_art_only pc_rep_only random
