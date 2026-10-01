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
# GDAL's NumPy module (osgeo.gdal_array) supports NumPy 2 only from GDAL 3.9.
# With an older system GDAL (Ubuntu 22.04: 3.4, 24.04: 3.8) it builds against
# NumPy 2 but fails to load, so keep NumPy 1.26 there (wheels exist for
# Python 3.10-3.12) and an OpenCV that still supports it.
if python -c "import sys; v = tuple(int(x) for x in '$GDAL_VERSION'.split('.')[:2]); sys.exit(0 if v < (3, 9) else 1)"; then
    if [ "$PY_MINOR" -le 12 ]; then
        echo "GDAL $GDAL_VERSION predates NumPy 2 support: using NumPy 1.26."
        pip install "numpy>=1.26,<2" "opencv-python-headless<4.12"
    else
        echo "WARNING: GDAL $GDAL_VERSION does not support NumPy 2, and Python 3.$PY_MINOR needs it."
        echo "         Co-registration will be unavailable (change detection still works)."
    fi
fi

echo "Installing GDAL Python bindings (v$GDAL_VERSION)..."
pip install --no-cache-dir --no-binary GDAL --no-build-isolation "GDAL==$GDAL_VERSION"

# The plugin itself works without osgeo.gdal_array (it reads and writes rasters
# through plain buffers), but co-registration (AROSICS) needs it. Rebuild once
# if it does not load, e.g. bindings built earlier against another NumPy.
if ! python -c "from osgeo import gdal_array" 2>/dev/null; then
    echo "GDAL's NumPy support does not load; rebuilding the bindings..."
    pip install --force-reinstall --no-deps --no-cache-dir --no-binary GDAL         --no-build-isolation "GDAL==$GDAL_VERSION" >/dev/null 2>&1 || true
fi
if python -c "from osgeo import gdal_array" 2>/dev/null; then
    echo "GDAL NumPy support: OK"
else
    echo "WARNING: GDAL's NumPy support (osgeo.gdal_array) does not load:"
    { python -c "import osgeo._gdal_array" 2>&1 || true; } | tail -1 | sed 's/^/         /'
    echo "         Change detection works, but automatic co-registration will be skipped."
fi

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

GITHUB_RELEASE="https://github.com/ja1902/ChangesDetector/releases/download/v0.8.0"

# A failed download warns and continues: the rest of the install still runs,
# and the file can be placed by hand afterwards.
download_weights() {
    local file="$1"
    local name="$2"
    local dest="$SCRIPT_DIR/$file"
    if [ -f "$dest" ]; then
        echo "$name weights already exist, skipping download."
        return 0
    fi
    echo "Downloading $name..."
    local ok=0
    if command -v curl &>/dev/null; then
        curl -L --fail --progress-bar -o "$dest" "$GITHUB_RELEASE/$file" && ok=1
    elif command -v wget &>/dev/null; then
        wget --show-progress -O "$dest" "$GITHUB_RELEASE/$file" && ok=1
    else
        echo "ERROR: Neither curl nor wget found. Please install one."
    fi
    if [ "$ok" -ne 1 ]; then
        rm -f "$dest"
        echo "WARNING: Download failed. Please download $file from"
        echo "         https://github.com/ja1902/ChangesDetector/releases and place it at:"
        echo "         $dest"
    fi
    return 0
}

# Model weights are for research / non-commercial use (CC BY-NC-SA 4.0 training data).
download_weights dinov2_vitb14_c2s1_levir.pth "DINOv2 ViT-B/14 + synthetic data (recommended)"
download_weights landcover_dinov2_vitb14_oem_second.pth "DINOv2 land-cover head (labelled change)"
download_weights dinov2_vitb14_levir.pth "DINOv2 ViT-B/14 (generalizable)"
download_weights dinov2_vitb14_egybcd.pth "DINOv2 ViT-B/14 (fine-tuned)"

# The "Changen2 ViT-L + DINOv2" model needs torchange (about 1 GB of extra
# packages plus ~1.2 GB of weights, CC BY-NC-SA 4.0). Installed by default;
# skip it with: INSTALL_CHANGEN2=0 ./install.sh
echo ""
if [[ ! "${INSTALL_CHANGEN2:-1}" =~ ^(0|[Nn]|no|false)$ ]]; then
    echo "Installing Changen2 model support (research use, ~2 GB)..."
    pip install --no-deps torchange ever-beta
    pip install einops timm albumentations tifffile tqdm wandb prettytable tensorboard matplotlib datasets huggingface_hub
    # same import as the plugin (includes its Python 3.10 compatibility step)
    # (loads only the bridge file, so a GDAL problem cannot mask the result)
    CHECK_CHANGEN2="import importlib.util as u; s = u.spec_from_file_location('changestar_bridge', '$SCRIPT_DIR/uchange_qgis_plugin/changestar_bridge.py'); m = u.module_from_spec(s); s.loader.exec_module(m); m.import_changen2()"
    if python -c "$CHECK_CHANGEN2" 2>/dev/null; then
        echo "Changen2 support: OK"
        # fetch the weights now, from huggingface.co, so the plugin never has to download at run time
        echo "Downloading Changen2 weights (~1.2 GB) from huggingface.co..."
        if python -c "import importlib.util as u; s = u.spec_from_file_location('changestar_bridge', '$SCRIPT_DIR/uchange_qgis_plugin/changestar_bridge.py'); m = u.module_from_spec(s); s.loader.exec_module(m); m._ensure_strenum(); print(m.prefetch_weights())"; then
            echo "Changen2 weights: OK"
        else
            echo "WARNING: Changen2 weights could not be downloaded (is huggingface.co reachable? a proxy or"
            echo "         firewall may block it). Re-run the installer once it is, or the plugin will retry on first use."
        fi
    else
        echo "WARNING: torchange does not import:"
        { python -c "$CHECK_CHANGEN2" 2>&1 || true; } | tail -1 | sed 's/^/         /'
    fi
else
    echo "Skipped Changen2 (INSTALL_CHANGEN2=0). The Changen2 model will show install instructions if selected."
fi

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
if [ -L "$PLUGIN_LINK" ]; then
    rm -f "$PLUGIN_LINK"
elif [ -d "$PLUGIN_LINK" ]; then
    # an earlier copied (not linked) install: keep it aside rather than delete it
    mv "$PLUGIN_LINK" "$PLUGIN_LINK.old.$(date +%Y%m%d%H%M%S)"
    echo "Moved an earlier plugin copy aside: $PLUGIN_LINK.old.*"
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
