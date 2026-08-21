#!/bin/bash
#SBATCH --job-name=signal_env
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS_GPU
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --output=logs/mkenv_%j.out
#
# Builds the conda environment. Set ENV to the prefix you want, for example
#   export ENV=$HOME/.conda/envs/posecover
# Run this on a GPU node if you want the CUDA build of torch selected correctly.
set -euo pipefail
: "${ENV:?set ENV to the conda environment prefix to create}"
module load conda   # site specific, omit if conda is already on PATH

conda create -y -p "$ENV" python=3.11

# torch first, so the CUDA wheel is resolved before anything pins numpy
conda run -p "$ENV" pip install --no-input torch --index-url https://download.pytorch.org/whl/cu128

conda run -p "$ENV" pip install --no-input \
    numpy scipy scikit-learn mediapipe opencv-python-headless \
    matplotlib tqdm sacrebleu

echo "=== verify ==="
conda run -p "$ENV" python -c "
import numpy, sklearn, cv2, torch, mediapipe
from mediapipe.tasks.python import vision
print('python      ', __import__('sys').version.split()[0])
print('numpy       ', numpy.__version__)
print('torch       ', torch.__version__, 'cuda', torch.cuda.is_available())
print('mediapipe   ', mediapipe.__version__)
print('opencv      ', cv2.__version__)
print('HolisticLandmarker present:', hasattr(vision, 'HolisticLandmarker'))
"
