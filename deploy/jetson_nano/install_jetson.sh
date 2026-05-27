#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Updating package index"
sudo apt-get update

echo "[2/4] Installing system packages"
sudo apt-get install -y \
  python3-pip \
  python3-venv \
  python3-dev \
  libopenblas-dev \
  libopenmpi-dev \
  libjpeg-dev \
  cmake

echo "[3/4] Installing Python packages"
python3 -m pip install --upgrade pip
python3 -m pip install \
  numpy \
  pillow \
  pandas \
  opencv-python \
  tqdm

echo "[4/4] Install complete"
echo "Install Jetson-compatible PyTorch and TorchVision according to the JetPack version in use."
