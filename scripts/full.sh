#!/bin/bash
#SBATCH --job-name=pc_full_spec
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS_GPU
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --output=logs/fullspec_%j.out

# Cluster specific settings. Adjust the SBATCH lines for your scheduler, then set:
#   PROJECT  directory holding src/, data/, poses/ and runs/
#   ENV      conda environment prefix created by mkenv.sh
: "${PROJECT:?set PROJECT to the project root}"
: "${ENV:?set ENV to the conda environment prefix}"

set -uo pipefail
module load conda   # site specific
ROOT=$PROJECT/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T
run () { conda run --no-capture-output -p $ENV python $PROJECT/src/run_budget.py \
           --root $ROOT --poses $PROJECT/poses --annotation translation "$@" ; }

# Budgets are 5/10/20% of the 9.19h pool at translation rates (impl. note 7).
B="2.8 5.5 11.0"

echo "########## 1. SPEC descriptors, WER, full baseline set ##########"
# The professor's protocol end to end: six descriptor groups at the specified codebook
# sizes, gloss recognition, subset-only vocabulary, every baseline from note 8.
run --out $PROJECT/runs/spec_wer --task gloss_ctc --descriptors spec \
    --budgets-hours $B --seeds 0 1 2 --max-epochs 60 \
    --strategies random source_random kcenter \
                 pc_rep_only pc_art_only pc_art_rep pc_full

echo "########## 2. privileged upper bounds (diagnostic only) ##########"
run --out $PROJECT/runs/spec_wer --task gloss_ctc --descriptors spec \
    --budgets-hours $B --seeds 0 --max-epochs 60 \
    --strategies privileged_text_fl privileged_oracle_gloss

echo "########## 3. finish the translation/BLEU seeds ##########"
run --out $PROJECT/runs/pcfull --budgets-hours 0.5 1 2 5 --archs gloss_free \
    --seeds 1 2 --n-handshape-clusters 400 --max-epochs 60 \
    --strategies random pc_rep_only pc_art_only pc_art_rep pc_full

echo "########## 4. resume the legacy-descriptor WER grid ##########"
run --out $PROJECT/runs/ctc --task gloss_ctc --budgets-hours $B \
    --seeds 0 1 2 --n-handshape-clusters 400 --max-epochs 60 \
    --strategies random pc_rep_only pc_art_only pc_art_rep pc_full
echo "ALL DONE"
