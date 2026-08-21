#!/bin/bash
#SBATCH --job-name=pc_tests
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS_CPU
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --output=logs/pctests_%j.out

# Cluster specific settings. Adjust the SBATCH lines for your scheduler, then set:
#   PROJECT  directory holding src/, data/, poses/ and runs/
#   ENV      conda environment prefix created by mkenv.sh
: "${PROJECT:?set PROJECT to the project root}"
: "${ENV:?set ENV to the conda environment prefix}"

module load conda   # site specific
conda run --no-capture-output -p $ENV \
  python $PROJECT/src/test_pc.py --poses $PROJECT/poses/train --code-dir $PROJECT/code
