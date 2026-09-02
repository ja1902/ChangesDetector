# Changes Detector

A research project exploring automated change detection between georeferenced satellite images, delivered as a QGIS plugin.

## What changed (v0.6)

CNN-based models like ChangerEx are **domain-locked** -- they achieve high F1 on in-domain data but fail on imagery from different regions or sensors. This version tackles that problem with a **DINOv2 vision transformer** for **generalizable change detection**. After 21 experiments (see [EXPERIMENTS.md](EXPERIMENTS.md)), the key finding is a fundamental trade-off between in-domain accuracy and cross-domain generalization: techniques that improve performance on the training domain consistently hurt generalization to unseen domains. Frozen DINOv2 features with a simple FPN decoder emerged as the best balance -- sacrificing a few points of in-domain F1 for reliable cross-domain performance.

### DINOv2 ViT-B/14

- **Frozen DINOv2 backbone + 4-layer FPN decoder**, trained on LEVIR-CD
- Two variants: **generalizable** (LEVIR-CD only) and **fine-tuned** (further tuned on another dataset). Fine-tuning improves accuracy on the target domain but hurts cross-domain generalization -- the same trade-off that applies at every level. The generalizable model is recommended for use on new/unseen regions. Try both to see which one performs better.
- Standalone ViT implementation -- runs in the QGIS plugin without heavy ML framework installs
- **Resolution-dynamic**: accepts any tile size (auto-crops to nearest multiple of 14). Default 256 becomes 252, giving ~4x fewer pixels per tile vs the fixed 518 that ViT-B/14 was trained on, with no accuracy loss
- DINOv2 is now the **default model** in both the QGIS plugin and CLI

### Auto-thresholding

- **Automatic threshold selection** enabled by default -- no manual tuning needed
- Models the unchanged distribution in log-probability space using a HWHM background model, sets threshold at 4.5 sigma above the unchanged peak
- Recovers 98-100% of oracle IoU across different change prevalences and cross-domain shifts
- Available via "Auto (recommended)" checkbox in QGIS and `--threshold auto` on the CLI

### Histogram matching

- Optional **per-tile histogram matching** to handle radiometric mismatch between image dates
- Normalizes each tile of the "after" image to match the "before" image's color distribution
- Helps when images have different brightness, contrast, or color balance due to different capture conditions
- Available via "Match image histograms" checkbox in QGIS and `--histogram-match` on the CLI

### Plugin UX improvements

- **Threshold slider** (0.00-1.00) replaces the old spin box, with auto mode as default
- **Default overlap** set to 32px for smoother tile boundaries
- **Small hole removal** in output polygons -- interior holes smaller than the min-area filter are cleaned up automatically

### DINOv2 training script

- `train_dinov2_cd.py` -- standalone DINOv2 training pipeline, no OpenCD/mmengine dependency
- Supports ViT-S/14, ViT-B/14, and ViT-L/14 backbones
- Trains only the decoder (~9M params) with the backbone frozen -- requires ~4GB VRAM
- Alternative to the OpenCD-based `finetune.py` -- no mmcv/mmengine dependency needed


### Key findings from 21 experiments

1. **Frozen DINOv2 + simple FPN + CE loss is optimal.** Everything that improves in-domain (Dice/Focal loss, backbone unfreezing, heavy augmentation, BAN decoder) hurts cross-domain generalization.
2. **Satellite-pretrained backbones generalize worse**, not better. DINOv3 SAT, CrossEarth, DOFA, and AnySat all underperformed plain DINOv2.


### Included Models

| Model | Training Dataset | Architecture | Mode |
|-------|-----------------|--------------|------|
| **DINOv2 ViT-B/14 (generalizable)** | LEVIR-CD | Frozen ViT-B/14 + FPN decoder | Binary CD |
| DINOv2 ViT-B/14 (fine-tuned) | LEVIR-CD + domain data | Frozen ViT-B/14 + FPN decoder | Binary CD |
| ChangerEx (R18) | LEVIR-CD | ResNet-18 + FDAF | Binary CD |
| SCD UPerNet (R18) | SECOND | UPerNet + ResNet-18 | Semantic CD |

## What changed (v0.5)

This version makes the plugin **compatible with Python 3.10 through 3.14**, so it works out of the box on both current Ubuntu LTS releases (22.04 with Python 3.10) and the latest (26.04 with Python 3.14). It also removes the dependency on the MMlab ecosystem (mmcv, mmseg, mmengine, open-cd) for inference.

### Why remove MMlab?

The original architecture used OpenCD/MMlab as the model framework. This worked but created significant friction:

- **mmcv compiles CUDA C++ extensions at install time**, which must exactly match your PyTorch + CUDA versions. This was the #1 source of install failures.
- **MMlab packages lag behind Python releases** -- they typically take months to support new versions. With Python 3.14 shipping as the default in Ubuntu 26.04, users would be stuck waiting.
- **Version conflicts** -- mmcv 2.x requires mmseg 1.x requires mmengine 0.x, and they all have to match exactly.
- **Size** -- mmcv + mmseg + opencd added 1-2 GB on top of PyTorch.

The model code (ChangerEx and SCD UPerNet) is now self-contained within the plugin -- pure PyTorch with no framework dependencies. This cuts the install to just `pip install torch` plus standard scientific Python packages.

### Subprocess-based inference

Previously the QGIS plugin imported PyTorch directly inside QGIS's own Python process. This broke when the venv's Python version didn't match QGIS's Python (e.g. a Python 3.14 venv on a system where QGIS uses Python 3.10 -- the compiled C extensions are incompatible).

Inference now runs as a **subprocess** via `detect_changes.py`, using the venv's own Python interpreter. QGIS's Python never loads PyTorch -- it just launches the subprocess and streams JSON progress back to the UI. This means the plugin works regardless of version mismatch between QGIS's Python and the venv's Python.

### Installer improvements

- `install.sh` now detects Python 3.10-3.14 and automatically selects the correct PyTorch CUDA variant (cu121 for Python ≤3.12, cu128 for Python 3.13+)
- Checks for missing system packages (`python3-venv`, `libgdal-dev`, `build-essential`) and tells the user exactly what to install
- Version pins relaxed for numpy and scipy so pre-built wheels are available on all supported Python versions

## What changed (v0.4)

This version adds **Semantic Change Detection (SCD)** as a selectable mode alongside the existing binary change detection. Instead of just detecting *where* change occurred, SCD classifies *what* the changed areas became -- water, ground, low vegetation, tree, building, or sports field.

### Semantic Change Detection

- New "Detection mode" selector in the plugin: **Binary Change Detection** or **Semantic Change Detection**
- SCD uses a **SiamEncoder-MultiDecoder (UPerNet + ResNet-18)** trained on the [SECOND dataset](https://captain-whu.github.io/SCD/) (6 land-cover classes)
- Outputs two GeoTIFF layers:
  - **Binary Change** -- change/no-change mask
  - **Semantic Change** -- land-cover classification of changed areas, transparent over unchanged areas so the satellite image shows through
- Colour-coded legend in QGIS with class names (water, ground, low vegetation, tree, building, sports field)
- CLI support: `python detect_changes.py --before img1.tif --after img2.tif --mode semantic`
- Co-registration, tiled inference, and GPU acceleration all work with SCD

## What changed (v0.3)

This version adds **automatic image co-registration** using [AROSICS](https://github.com/GFZ/arosics), which corrects sub-pixel spatial misalignment between image pairs before change detection. Satellite images captured at different times often have small GPS/sensor offsets that produce false change detections along edges and boundaries. Co-registration eliminates this noise.

### Co-registration

- Integrated AROSICS global shift correction into both the QGIS plugin and the standalone CLI
- Activates automatically when input images are georeferenced TIFFs with a valid CRS
- Detects and corrects shifts up to 50px (configurable via `--max-shift`)
- CRS compatibility pre-check with a clear error message if images need reprojection
- Handles images with invalid nodata metadata (e.g. nodata=256 on uint8 bands) that would otherwise crash AROSICS

### Tested impact

On a real-world New Zealand 20cm aerial image pair (2012 vs 2016, 11265x15354px):

| Scenario | Change detected |
|----------|----------------|
| Without co-registration | 2.70% |
| **With co-registration** | **2.04%** |

The ~0.7% difference is false positives caused by a 1.4px natural misalignment between captures. With a synthetic 10px shift applied, co-registration fully recovers the correct baseline (2.04%).


## What changed (v0.2)

The first version of this plugin shipped with **MambaBCD** (a state-space model) and **PeftCD** (DINOv3 + LoRA), which I selected based on their published benchmark scores and my own testing. I had looked at the models in the [Open-CD](https://github.com/likyoo/open-cd) repository but dismissed them -- they were older CNN architectures with similar reported F1 scores, and I assumed the newer approaches would be faster and more practical at inference time.

However, after running a proper 18-model benchmark on the same hardware, **ChangerEx (R18)** -- a straightforward ResNet-18 Siamese encoder-decoder from Open-CD -- turned out to be dramatically faster and lighter than both MambaBCD and PeftCD, while matching them on accuracy. This version replaces both models with ChangerEx.

### Why ChangerEx?

ChangerEx uses a ResNet-18 backbone. Despite being simpler and older than the models it replaces, it dominates on the accuracy-efficiency tradeoff:

| Model | F1 | Time (ms) | VRAM (MB) |
|-------|-----|-----------|-----------|
| **ChangerEx (R18)** | **0.918** | **59** | **448** |
| PeftCD (DINOv3+LoRA) | 0.915 | 1,891 | 4,622 |
| MambaBCD (VMamba) | 0.907 | 5,190 | 6,401 |

- **30x faster** than PeftCD, **88x faster** than MambaBCD
- **10x less VRAM** than PeftCD, **14x less** than MambaBCD
- F1 within 0.3% of the best model tested (CGNet, 0.921)

For full benchmark results and analysis, see [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md).

### Findings on generalization

A central finding is that most change detection models are **domain-locked** -- they perform well on imagery similar to their training set but struggle on anything else. **The only reliable path to accurate results on a specific area is fine-tuning on labelled data from that region using the same image source.**

---

## QGIS Plugin

### Included Models

| Model | Training Dataset | Architecture | Mode |
|-------|-----------------|--------------|------|
| **DINOv2 ViT-B/14 (generalizable)** | LEVIR-CD | Frozen ViT-B/14 + FPN decoder | Binary CD |
| DINOv2 ViT-B/14 (fine-tuned) | LEVIR-CD + domain data | Frozen ViT-B/14 + FPN decoder | Binary CD |
| ChangerEx (R18) | LEVIR-CD | ResNet-18 + FDAF | Binary CD |
| SCD UPerNet (R18) | SECOND | UPerNet + ResNet-18 | Semantic CD |

### Prerequisites

- **Python 3.10-3.14** (Ubuntu 22.04-26.04 all supported)
- **QGIS 3.22+** (any Python version -- the plugin runs inference in a subprocess)
- **NVIDIA GPU** (recommended, CPU also supported)
- **System packages**: `sudo apt install python3-venv python3-dev libgdal-dev build-essential`

### Installation

Clone the repository and run the installer:

```bash
git clone https://github.com/ja1902/ChangesDetector.git
cd ChangesDetector
```

**Linux:**
```bash
chmod +x install.sh
./install.sh
```

**Windows:**
```
install.bat
```

The installer will:
1. Create a Python virtual environment
2. Install PyTorch (with CUDA if GPU detected, CPU otherwise)
3. Install all dependencies
4. Download model weights
5. Link the plugin into your QGIS plugins directory

### Usage

1. Open QGIS
2. Go to **Plugins > Manage and Install Plugins**
3. Enable **"ChangeDetection"**
4. Open the plugin from **Plugins > ChangeDetection**
5. Select your **before** and **after** raster layers
6. Choose detection mode: **Binary Change Detection** or **Semantic Change Detection**
7. Select device: **Auto**, **CPU**, or **GPU**
8. Set processing parameters (tile size, overlap, threshold for binary mode)
9. Choose an output path (GeoPackage for binary, GeoTIFF for semantic)
10. Click **Run**

### Manual Weight Download

If the installer cannot download weights automatically, download them from the [GitHub Releases page](https://github.com/ja1902/ChangesDetector/releases) and place in the project root:

- `dinov2_vitb14_levir.pth` (364MB) -- DINOv2 ViT-B/14 generalizable binary CD
- `dinov2_vitb14_egybcd.pth` (365MB) -- DINOv2 ViT-B/14 fine-tuned binary CD
- `ChangerEx_r18-512x512_40k_levircd.pth` -- ChangerEx binary CD
- `scd_upernet_r18_10k_second.pth` -- SCD semantic CD

### Standalone CLI

```bash
# Binary change detection with DINOv2 (auto threshold)
python detect_changes.py --before path/to/before.tif --after path/to/after.tif \
    --model-type dinov2 --weights dinov2_vitb14_levir.pth --threshold auto

# With histogram matching (for mismatched image pairs)
python detect_changes.py --before path/to/before.tif --after path/to/after.tif \
    --model-type dinov2 --weights dinov2_vitb14_levir.pth --threshold auto --histogram-match

# Binary change detection with ChangerEx
python detect_changes.py --before path/to/before.tif --after path/to/after.tif

# Semantic change detection
python detect_changes.py --before path/to/before.tif --after path/to/after.tif --mode semantic

# Output as GeoPackage polygons
python detect_changes.py --before path/to/before.tif --after path/to/after.tif \
    --model-type dinov2 --weights dinov2_vitb14_levir.pth \
    --threshold auto --output-gpkg changes.gpkg --min-area 100
```

Options: `--mode binary|semantic`, `--model-type opencd|dinov2`, `--threshold auto|0.3`, `--histogram-match`, `--overlap 32`, `--tile-size 256`, `--weights path/to/weights.pth`, `--no-coreg`, `--max-shift 50`, `--coreg-window 1024`

### Training

`train_dinov2_cd.py` trains DINOv2 decoders on standard change detection datasets (LEVIR-CD format: `train/A/`, `train/B/`, `train/label/`).

```bash
# Train on LEVIR-CD
python train_dinov2_cd.py --data-root /path/to/LEVIR-CD --backbone vitb14 --epochs 50

# Fine-tune on a custom dataset
python train_dinov2_cd.py --data-root /path/to/custom-dataset \
    --backbone vitb14 --weights work_dirs/decoder_best.pth --epochs 20 --lr 1e-4
```

The backbone stays frozen; only the decoder trains (~9M params, ~4GB VRAM).

## Troubleshooting

**Plugin doesn't appear in QGIS:**
- Check that the plugin is enabled in Plugin Manager
- Verify the symlink/junction exists in your QGIS plugins directory
- Restart QGIS after installation

**"Model weights not found" error:**
- Run the installer to download weights, or download manually (see above)
- Or use "Use custom weights file" to point to your own checkpoint

**"CUDA GPU is not available" error:**
- Select "CPU" as the device, or install NVIDIA drivers + CUDA toolkit

**GPU out of memory:**
- Reduce tile size (e.g. `--tile-size 128`) or use `--device cpu`
- The tiler automatically adjusts batch size and retries on OOM

**Slow inference:**
- Use GPU if available
- Increase tile size (256 -> 512) to process fewer tiles
- Reduce overlap (64 -> 0) for faster but slightly less smooth results
