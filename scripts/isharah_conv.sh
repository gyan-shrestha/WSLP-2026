#!/bin/bash
#SBATCH --job-name=isharah_conv
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS_GPU
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --output=logs/isharah_conv_%j.out

# Cluster specific settings. Adjust the SBATCH lines for your scheduler, then set:
#   PROJECT  directory holding src/, data/, poses/ and runs/
#   ENV      conda environment prefix created by mkenv.sh
: "${PROJECT:?set PROJECT to the project root}"
: "${ENV:?set ENV to the conda environment prefix}"

set -euo pipefail
module load conda   # site specific
conda run --no-capture-output -p $ENV \
  python $PROJECT/src/isharah_to_npz.py \
    --pkl $PROJECT/data/isharah/pose_data_isharah2000_hands_lips_body.pkl \
    --out $PROJECT/poses_isharah
