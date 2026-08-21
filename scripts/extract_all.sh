#!/bin/bash
#SBATCH --job-name=pose_all
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS_GPU
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12gb
#SBATCH --array=0-39
#SBATCH --output=logs/pose_%A_%a.out

# Cluster specific settings. Adjust the SBATCH lines for your scheduler, then set:
#   PROJECT  directory holding src/, data/, poses/ and runs/
#   ENV      conda environment prefix created by mkenv.sh
: "${PROJECT:?set PROJECT to the project root}"
: "${ENV:?set ENV to the conda environment prefix}"

set -euo pipefail
module load conda   # site specific
for SPLIT in train dev test; do
  conda run --no-capture-output -p $ENV python $PROJECT/src/extract_pose.py \
      --root $PROJECT/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
      --out $PROJECT/poses --model $PROJECT/src/holistic_landmarker.task \
      --split $SPLIT --shard $SLURM_ARRAY_TASK_ID --n-shards 40 2>/dev/null
done
echo "[shard $SLURM_ARRAY_TASK_ID] ALL SPLITS DONE"
