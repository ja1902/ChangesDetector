#!/bin/bash
# Setup script for the CD benchmark on an offline Linux machine.
# Run once before benchmark_cd.py.
#
# Usage:
#   chmod +x setup_benchmark.sh
#   ./setup_benchmark.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "============================================"
echo "  CD Benchmark Setup"
echo "============================================"

# Create venv
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
else
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
echo "Python: $(which python)"
echo "Version: $(python --version)"

# Upgrade pip and pin setuptools
# setuptools 82+ removed pkg_resources (breaks mmcv build)
# setuptools <68 is too old for modern builds
# torch requires setuptools <82
pip install --upgrade pip
pip install "setuptools==75.8.0" wheel

# Install PyTorch (assumes CUDA is available — adjust if needed)
echo ""
echo "Installing PyTorch..."
pip install torch torchvision

# Install Open-CD and its dependencies
# Order matters: setuptools → mmengine → mmcv → mmseg → open-cd
echo ""
echo "Installing OpenMMLab stack..."
pip install -U openmim
mim install mmengine

# mmcv: prebuilt wheels only exist for common CUDA/torch combos.
# If mim fails, fall back to building from source.
echo ""
echo "Installing mmcv (may build from source — takes a few minutes)..."
mim install "mmcv>=2.0.0" || pip install "mmcv>=2.0.0" --no-build-isolation

pip install "mmsegmentation>=1.2.2"
pip install "mmdet>=3.0.0"
mim install "mmpretrain>=1.0.0rc7"

echo ""
echo "Installing Open-CD..."
pip install -v -e "$SCRIPT_DIR"

# Install benchmark dependencies
echo ""
echo "Installing benchmark dependencies..."
pip install scipy opencv-python-headless

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To run the benchmark:"
echo "  source $VENV_DIR/bin/activate"
echo "  python benchmark_cd.py --data-root /path/to/LEVIR-CD/test"
echo ""
