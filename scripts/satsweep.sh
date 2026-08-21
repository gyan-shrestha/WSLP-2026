#!/bin/bash
#SBATCH --job-name=sat_sweep
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS_GPU
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --output=logs/sat_%j.out

# Cluster specific settings. Adjust the SBATCH lines for your scheduler, then set:
#   PROJECT  directory holding src/, data/, poses/ and runs/
#   ENV      conda environment prefix created by mkenv.sh
: "${PROJECT:?set PROJECT to the project root}"
: "${ENV:?set ENV to the conda environment prefix}"

set -euo pipefail
module load conda   # site specific
for NHS in 40 150 400; do
  echo ""
  echo "############### n_handshape_clusters = $NHS ###############"
  conda run --no-capture-output -p $ENV python $PROJECT/src/run_budget.py \
    --root $PROJECT/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --poses $PROJECT/poses --out $PROJECT/runs/sat_nhs$NHS \
    --budgets-hours 0.5 1 2 5 10 20 40 \
    --strategies random duration_random phon_coverage \
    --annotation translation --seeds 0 \
    --n-handshape-clusters $NHS --select-only 2>&1 | grep -vE "UserWarning|self\.|warnings"
done
