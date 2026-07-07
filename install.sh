#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " ChangeDetection QGIS Plugin Installer (Linux)"
echo "============================================"
echo ""

# -----------------------------------------------
# 1. Check prerequisites
# -----------------------------------------------
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.10+."
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Found Python $PY_VER"

# -----------------------------------------------
# 2. Create virtual environment
# -----------------------------------------------
VENV_DIR="$SCRIPT_DIR/venv"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
else
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# -----------------------------------------------
# 3. Install PyTorch (GPU or CPU)
# -----------------------------------------------
echo ""
echo "Detecting GPU..."
if command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No NVIDIA GPU detected. Installing PyTorch CPU-only..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# -----------------------------------------------
# 4. Install Python dependencies
# -----------------------------------------------
echo ""
echo "Installing dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# mmcv requires a CUDA-specific wheel
echo "Installing mmcv..."
if command -v nvidia-smi &>/dev/null; then
    pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
else
    pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.4/index.html
fi

# -----------------------------------------------
# 5. Download model weights from GitHub Releases
# -----------------------------------------------
echo ""
echo "Downloading model weights..."

GITHUB_RELEASE="https://github.com/ja1902/ChangesDetector/releases/download/v0.2.0"

download_weights() {
    local url="$1"
    local dest="$2"
    local name="$3"
    if [ -f "$dest" ]; then
        echo "$name weights already exist, skipping download."
        return 0
    fi
    echo "Downloading $name..."
    if command -v curl &>/dev/null; then
        curl -L --fail --progress-bar -o "$dest" "$url"
    elif command -v wget &>/dev/null; then
        wget --show-progress -O "$dest" "$url"
    else
        echo "ERROR: Neither curl nor wget found. Please install one."
        return 1
    fi
    if [ $? -ne 0 ]; then
        rm -f "$dest"
        echo "WARNING: Download failed. Please download manually and place at:"
        echo "         $dest"
    fi
}

download_weights \
    "$GITHUB_RELEASE/ChangerEx_r18-512x512_40k_levircd.pth" \
    "$SCRIPT_DIR/ChangerEx_r18-512x512_40k_levircd.pth" \
    "ChangerEx R18 (LEVIR-CD)"

# -----------------------------------------------
# 6. Write environment config for plugin
# -----------------------------------------------
echo ""
echo "Writing environment config..."
VENV_SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
cat > "$SCRIPT_DIR/uchange_qgis_plugin/_env_config.py" <<PYEOF
VENV_SITE_PACKAGES = "$VENV_SP"
PYEOF
echo "  Written: _env_config.py"

# -----------------------------------------------
# 7. Install plugin to QGIS
# -----------------------------------------------
echo ""
QGIS_PLUGINS="$HOME/.local/share/QGIS/QGIS3/profiles/default/python/plugins"
mkdir -p "$QGIS_PLUGINS"

PLUGIN_LINK="$QGIS_PLUGINS/uchange_qgis_plugin"
if [ -L "$PLUGIN_LINK" ] || [ -d "$PLUGIN_LINK" ]; then
    rm -f "$PLUGIN_LINK"
fi
ln -s "$SCRIPT_DIR/uchange_qgis_plugin" "$PLUGIN_LINK"
echo "Plugin symlinked to: $PLUGIN_LINK"

# -----------------------------------------------
# Done
# -----------------------------------------------
echo ""
echo "============================================"
echo " Installation complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Open QGIS"
echo "  2. Go to Plugins > Manage and Install Plugins"
echo "  3. Enable 'ChangeDetection'"
echo "  4. Find it under Plugins > ChangeDetection menu"
echo ""
