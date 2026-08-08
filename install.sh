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
MISSING_PKGS=()

PYTHON_CMD=""
for candidate in python3 python3.14 python3.13 python3.12 python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        if "$candidate" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)" 2>/dev/null; then
            PYTHON_CMD="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.10+ not found."
    echo ""
    echo "Install Python 3.10 or newer and try again."
    exit 1
fi

PY_VER=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Found Python $PY_VER ($PYTHON_CMD)"

if ! $PYTHON_CMD -m venv --help &>/dev/null; then
    MISSING_PKGS+=("python${PY_VER}-venv")
fi

if ! $PYTHON_CMD -c "import distutils" 2>/dev/null && ! $PYTHON_CMD -c "import sysconfig" 2>/dev/null; then
    MISSING_PKGS+=("python${PY_VER}-dev")
fi

if ! command -v gcc &>/dev/null; then
    MISSING_PKGS+=("build-essential")
fi

if ! command -v gdal-config &>/dev/null; then
    MISSING_PKGS+=("libgdal-dev")
fi

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo ""
    echo "Missing system packages: ${MISSING_PKGS[*]}"
    read -rp "Install them now? (requires sudo) [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        sudo apt update && sudo apt install -y ${MISSING_PKGS[*]}
    else
        echo ""
        echo "Install them manually with:"
        echo "  sudo apt update && sudo apt install ${MISSING_PKGS[*]}"
        exit 1
    fi
fi

GDAL_VERSION=$(gdal-config --version)
echo "Found GDAL $GDAL_VERSION"

# -----------------------------------------------
# 2. Create virtual environment
# -----------------------------------------------
VENV_DIR="$SCRIPT_DIR/venv"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
else
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# -----------------------------------------------
# 3. Install PyTorch (GPU or CPU)
# -----------------------------------------------
echo ""
echo "Detecting GPU..."

PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    if [ "$PY_MINOR" -ge 13 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    fi
    pip install torch torchvision --index-url "$TORCH_INDEX"
else
    echo "No NVIDIA GPU detected. Installing PyTorch CPU-only..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# -----------------------------------------------
# 4. Install Python dependencies
# -----------------------------------------------
echo ""
echo "Installing dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Pin GDAL Python bindings to match the system library version.
# Build from source (--no-binary) with numpy present (--no-build-isolation)
# so that gdal_array is compiled correctly.
echo "Installing GDAL Python bindings (v$GDAL_VERSION)..."
pip install --no-cache-dir --no-binary GDAL --no-build-isolation "GDAL==$GDAL_VERSION"

# arosics/geoarray/py_tools_ds officially require GDAL >= 3.8, but work
# with older versions via our compatibility shims in _gdal_compat.py.
# Install without deps to avoid pip pulling an incompatible GDAL version.
echo "Installing AROSICS (co-registration)..."
pip install --no-deps arosics geoarray py_tools_ds

# -----------------------------------------------
# 5. Download model weights from GitHub Releases
# -----------------------------------------------
echo ""
echo "Downloading model weights..."

GITHUB_RELEASE="https://github.com/ja1902/ChangesDetector/releases/download/v0.5.0"

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

download_weights \
    "$GITHUB_RELEASE/scd_upernet_r18_10k_second.pth" \
    "$SCRIPT_DIR/scd_upernet_r18_10k_second.pth" \
    "SCD UPerNet R18 (SECOND)"

# -----------------------------------------------
# 6. Write environment config for plugin
# -----------------------------------------------
echo ""
echo "Writing environment config..."
VENV_SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
VENV_PY="$VENV_DIR/bin/python"
cat > "$SCRIPT_DIR/uchange_qgis_plugin/_env_config.py" <<PYEOF
VENV_SITE_PACKAGES = "$VENV_SP"
VENV_PYTHON = "$VENV_PY"
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
