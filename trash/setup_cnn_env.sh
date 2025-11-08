#!/usr/bin/env bash
# setup_cnn_env.sh
# Usage: bash setup_cnn_env.sh
set -e

ENV_NAME="cnn"
PY_VER=3.10
CUDA_TOOLKIT="cpuonly"
# Edit the above as needed for CUDA.

echo "\n==== [PyTorch CIFAR-10 ENV SETUP] ===="
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi
if conda info --envs | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "[INFO] The conda env '$ENV_NAME' already exists."
else
    echo "[INFO] Creating conda env '$ENV_NAME'..."
    conda create -y -n $ENV_NAME python=$PY_VER
fi

echo "[INFO] Activating environment and installing packages..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install torch, torchvision, matplotlib, numpy
echo "[INFO] Installing torch, torchvision, matplotlib, numpy..."
conda install -y pip
pip install --upgrade pip
conda install -y matplotlib numpy
pip install torch torchvision

echo "\n[INFO] Environment setup complete!"
echo "----------------------------------------------------"
echo "To use your project, run:"
echo "   conda activate cnn"
echo "   python cifar10_tinycnn.py"
echo "----------------------------------------------------"
