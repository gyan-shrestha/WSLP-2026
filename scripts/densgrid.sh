#!/bin/bash
#SBATCH --job-name=dens_grid
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS_GPU
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --output=logs/densgrid_%j.out

# Cluster specific settings. Adjust the SBATCH lines for your scheduler, then set:
#   PROJECT  directory holding src/, data/, poses/ and runs/
#   ENV      conda environment prefix created by mkenv.sh
: "${PROJECT:?set PROJECT to the project root}"
: "${ENV:?set ENV to the conda environment prefix}"

set -uo pipefail
module load conda   # site specific

# Part 2 of the "why does coverage fail" experiment: remove the sparsest 10 and 20
# percent of candidates, then run exactly the same selection on what remains. Each
# filtered variant has an unfiltered counterpart already in runs/spec_wer, so the
# comparison is like for like.
echo "########## PHOENIX14T, density-filtered ##########"
conda run --no-capture-output -p $ENV python $PROJECT/src/run_budget.py \
  --root $PROJECT/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T --poses $PROJECT/poses \
  --out $PROJECT/runs/spec_wer --task gloss_ctc --descriptors spec \
  --annotation translation --budgets-hours 2.8 5.5 11.0 --seeds 0 1 2 \
  --max-epochs 60 \
  --strategies dens10_art_only dens20_art_only dens10_full dens20_full

echo "########## Isharah, density-filtered ##########"
conda run --no-capture-output -p $ENV python $PROJECT/src/run_budget.py \
  --corpus isharah --isharah-split SI \
  --root $PROJECT/data/isharah/annotations --poses $PROJECT/poses_isharah \
  --out $PROJECT/runs/isharah_wer --task gloss_ctc --descriptors spec \
  --annotation translation --budgets-hours 7.1 14.2 28.4 --seeds 0 1 2 \
  --max-epochs 60 \
  --strategies dens10_art_only dens20_art_only dens10_full dens20_full
echo "ALL DONE"
