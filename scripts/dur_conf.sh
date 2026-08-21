#!/bin/bash
#SBATCH --job-name=durconf
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS_CPU
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --output=logs/durconf_%j.out

# Cluster specific settings. Adjust the SBATCH lines for your scheduler, then set:
#   PROJECT  directory holding src/, data/, poses/ and runs/
#   ENV      conda environment prefix created by mkenv.sh
: "${PROJECT:?set PROJECT to the project root}"
: "${ENV:?set ENV to the conda environment prefix}"

set -uo pipefail
module load conda   # site specific
conda run --no-capture-output -p $ENV python $PROJECT/src/dur_conf.py \
  --root $PROJECT/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T --poses $PROJECT/poses \
  --corpus phoenix --density-json $PROJECT/runs/density_phoenix.json --k 20
conda run --no-capture-output -p $ENV python $PROJECT/src/dur_conf.py \
  --root $PROJECT/data/isharah/annotations --poses $PROJECT/poses_isharah \
  --corpus isharah --density-json $PROJECT/runs/density_isharah.json --k 20
echo ALL DONE
