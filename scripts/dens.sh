#!/bin/bash
#SBATCH --job-name=density
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS_CPU
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --output=logs/density_%j.out

# Cluster specific settings. Adjust the SBATCH lines for your scheduler, then set:
#   PROJECT  directory holding src/, data/, poses/ and runs/
#   ENV      conda environment prefix created by mkenv.sh
: "${PROJECT:?set PROJECT to the project root}"
: "${ENV:?set ENV to the conda environment prefix}"

set -uo pipefail
module load conda   # site specific
echo "########## PHOENIX14T, 10 percent budget ##########"
conda run --no-capture-output -p $ENV python $PROJECT/src/density.py \
  --root $PROJECT/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T --poses $PROJECT/poses \
  --corpus phoenix --budget 5.5 --k 20 --out $PROJECT/runs/density_phoenix.json
echo ""
echo "########## Isharah, 10 percent budget ##########"
conda run --no-capture-output -p $ENV python $PROJECT/src/density.py \
  --root $PROJECT/data/isharah/annotations --poses $PROJECT/poses_isharah \
  --corpus isharah --budget 14.2 --k 20 --out $PROJECT/runs/density_isharah.json
echo "ALL DONE"
