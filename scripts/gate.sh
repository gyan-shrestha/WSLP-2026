#!/bin/bash
#SBATCH --job-name=gate
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS_GPU
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --output=logs/gate_%j.out

# Cluster specific settings. Adjust the SBATCH lines for your scheduler, then set:
#   PROJECT  directory holding src/, data/, poses/ and runs/
#   ENV      conda environment prefix created by mkenv.sh
: "${PROJECT:?set PROJECT to the project root}"
: "${ENV:?set ENV to the conda environment prefix}"

set -euo pipefail
module load conda   # site specific
echo "################ DETECTION BY SIGNER ################"
conda run --no-capture-output -p $ENV python $PROJECT/src/det_by_signer.py $PROJECT/poses/train
echo ""
echo "################ WEEK-2 KILL GATE ################"
conda run --no-capture-output -p $ENV python $PROJECT/src/cluster_gate.py \
    --poses $PROJECT/poses/train --n-clips 1500 --n-handshape-clusters 40 \
    --json-out $PROJECT/runs/gate_phoenix.json
