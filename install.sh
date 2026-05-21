#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " ChangeDetection QGIS Plugin Installer (Linux)"
echo "============================================"
echo ""

PYTHON=""

for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        version="$($candidate -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        case "$version" in
            3.10|3.11|3.12)
                PYTHON="$candidate"
                break
                ;;
        esac
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10, 3.11, or 3.12 is required."
    echo "       Python 3.13 is not supported because mmcv does not provide a compatible wheel."
    echo "       Install one of the supported versions and rerun this script."
    exit 1
fi

# -----------------------------------------------
# 1. Check prerequisites
# -----------------------------------------------
PY_VER="$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
echo "Using Python executable: $PYTHON"
echo "Found Python $PY_VER"

# -----------------------------------------------
# 2. Create virtual environment
# -----------------------------------------------
VENV_DIR="$SCRIPT_DIR/venv"
RECREATE_VENV=0
if [ -d "$VENV_DIR" ]; then
    if [ -x "$VENV_DIR/bin/python" ]; then
        VENV_PY_VER="$($VENV_DIR/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
        if [ "$VENV_PY_VER" != "$PY_VER" ]; then
            echo "Virtual environment at $VENV_DIR uses Python $VENV_PY_VER, expected $PY_VER. Recreating it..."
            RECREATE_VENV=1
        else
            echo "Virtual environment already exists at $VENV_DIR"
        fi
    else
        echo "Virtual environment at $VENV_DIR is incomplete. Recreating it..."
        RECREATE_VENV=1
    fi
fi

if [ "$RECREATE_VENV" -eq 1 ]; then
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
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

GITHUB_RELEASE="https://github.com/ja1902/ChangesDetector/releases/download/v0.1.0"

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
        if ! curl -L --fail --progress-bar -o "$dest" "$url"; then
            rm -f "$dest"
            echo "WARNING: $name download failed."
            echo "         Please download manually and place it at:"
            echo "         $dest"
            return 0
        fi
    elif command -v wget &>/dev/null; then
        if ! wget --show-progress -O "$dest" "$url"; then
            rm -f "$dest"
            echo "WARNING: $name download failed."
            echo "         Please download manually and place it at:"
            echo "         $dest"
            return 0
        fi
    else
        echo "ERROR: Neither curl nor wget found. Please install one."
        echo "WARNING: Please download $name manually and place it at:"
        echo "         $dest"
    fi
}

download_weights \
    "$GITHUB_RELEASE/MambaBCD_Small_LEVIRCD+.pth" \
    "$SCRIPT_DIR/MambaBCD_Small_LEVIRCD+.pth" \
    "MambaBCD-Small (LEVIR-CD+, ~207 MB)"

download_weights \
    "$GITHUB_RELEASE/MambaBCD_Small_SYSU.pth" \
    "$SCRIPT_DIR/MambaBCD_Small_SYSU.pth" \
    "MambaBCD-Small (SYSU, ~207 MB)"

download_weights \
    "$GITHUB_RELEASE/PeftCD_LEVIRCD.ckpt" \
    "$SCRIPT_DIR/PeftCD_LEVIRCD.ckpt" \
    "PeftCD (LEVIR-CD+)"

download_weights \
    "$GITHUB_RELEASE/PeftCD_SYSU.ckpt" \
    "$SCRIPT_DIR/PeftCD_SYSU.ckpt" \
    "PeftCD (SYSU)"

# -----------------------------------------------
# 6. Write environment config for plugin
# -----------------------------------------------
echo ""
echo "Writing environment config..."
VENV_SP=$("$PYTHON" -c "import site; print(site.getsitepackages()[0])")
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
