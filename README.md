# Changes Detector

A research project exploring automated change detection between georeferenced satellite images, delivered as a QGIS plugin.

## What changed (v0.8)

### Labelled change: what changed into what

Semantic mode has a new recommended model, **DINOv2 ViT-B + land cover, from -> to**. The v0.7 DINOv2 model finds the changes; a land-cover head on the same frozen backbone (trained on OpenEarthMap, 44 countries, plus SECOND) classifies each date. Every change polygon gets one before class and one after class (the majority inside it) and is written to a GeoPackage with the fields `from_class`, `to_class`, `change` (e.g. "farmland -> building") and `area`. Classes: bare ground, grass / low vegetation, paved / road, trees, water, farmland, building. The layer is coloured by what each area became.

Share of truly changed pixels whose before *and* after class are right:

| | SECOND (China) | HRSCD (France, never seen) |
|---|---|---|
| SCD UPerNet (v0.7) | 60% | 40% |
| **DINOv2 + land cover (v0.8)** | **63%** | **50%** |

Changes are found by the building-focused v0.7 model, so this mode suits building and urban change: on the French data it found changes far better than the UPerNet (IoU 0.22 vs 0.05). Changes that involve no building, such as forest cleared for farmland, are largely missed. Needs `landcover_dinov2_vitb14_oem_second.pth` (the installer downloads it). CLI: `--model-type dinov2_lc --preset synthetic --threshold auto --output-gpkg out.gpkg`.

### Models dropped

ChangerEx (R18) and the SCD UPerNet (R18) are no longer installed or listed: the DINOv2 models beat ChangerEx on every unseen dataset tested, and the new labelled-change mode replaces the SCD UPerNet for building and urban change. Their weights stay on the [v0.6.0 release](https://github.com/ja1902/ChangesDetector/releases/tag/v0.6.0), and the CLI still runs them with `--model-type opencd` / `--model-type opencd_scd --mode semantic`. All v0.8 weights are on the single v0.8.0 release.

### Fixes

- **CPU**: DINOv2 models crashed on computers without an NVIDIA GPU (float16 weights, float32 inputs); they now run in float32 on CPU.
- **Installer**: a failed weight download no longer stops the installer. It now checks GDAL's NumPy support after building the bindings, and offers optional Changen2 support (`INSTALL_CHANGEN2=1 ./install.sh` for unattended installs).
- **GeoTIFFs on some Linux installs**: when GDAL's Python bindings were built without NumPy support (`No module named '_gdal_array'`), GeoTIFFs were silently read without georeferencing and writing polygons failed. Rasters are now read and written without that module, and a failed GDAL read is reported instead of hidden.
- **CLI defaults**: with no model given, `detect_changes.py` now runs the recommended DINOv2 model with its tested settings (it used to default to ChangerEx).

## What changed (v0.7)

v0.7 targets three problems: working on new regions, sensors and resolutions without retraining; false alarms on angled (off-nadir) imagery; and naming what changed into what.

### New default model: DINOv2 + synthetic change data

Same architecture and file format as the v0.6 DINOv2 model, but the decoder is trained on [Changen2](https://github.com/Z-Zheng/pytorch-change-models) synthetic change pairs plus LEVIR-CD (`dinov2_vitb14_c2s1_levir.pth`). IoU on test sets, with each model's threshold picked on a *different* unseen dataset (the fair setting for new imagery; v0.6's own auto-threshold scored lower, e.g. 0.20 on S2Looking):

| IoU | S2Looking | EGY-BCD | LEVIR-CD |
|---|---|---|---|
| v0.6 DINOv2 (generalizable) | 0.27 | 0.43 | 0.84 |
| **v0.7 DINOv2 + synthetic** | **0.35** | **0.55** | 0.76-0.80 |
| v0.7 Changen2 ViT-L + DINOv2 (optional) | **0.40** | 0.53 | 0.80 |

At the same detection rate it raises about 4x fewer false alarms than v0.6 on imagery of the same place seen from different angles (SpaceNet MVOI). Trained on CC BY-NC-SA 4.0 data: research / non-commercial use.

### Each model runs the way it was tested

Models now carry a preset: 1.4x upscaling, flip averaging, logit adjustment and a threshold chosen on imagery the model had not seen. The "Change threshold" box is now **Recommended**: for preset models it uses that tested threshold, not the per-scene estimate (which reached only ~70% of the best possible IoU on unfamiliar data). CLI: `--preset synthetic` or `--preset ensemble` with `--threshold auto`.

### Coarse imagery is resampled automatically

Every model collapses on imagery much coarser than it was trained on (2 m LEVIR: IoU about 0.1). Rasters whose pixel size is coarser than 0.5 m are now resampled to 0.5 m before detection, and results are written at the original resolution. On 2 m test GeoTIFFs this took the new model from 0.39 to 0.73 IoU. CLI: `--target-gsd 0.5` (0 to turn off).

### Optional: Changen2 ViT-L + DINOv2 (most accurate)

Runs the Changen2 ChangeStar ViT-L model alongside the DINOv2 model and keeps only change both agree on. It needs the optional `torchange` package (the plugin prints install instructions) and is slower; ChangeStar weights are CC BY-NC-SA 4.0.

### Fixes

- **Semantic change legend**: the SCD UPerNet's outputs were shown with the wrong class names (low vegetation as "water", tree as "low vegetation", water as "tree"). With the corrected order its SeK on SECOND rises from 0.13 to 0.20.
- **Newer GPUs**: DINOv2 models failed on RTX 30-series and newer (float16 weights under bfloat16 autocast); autocast now follows the model's precision.

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
| **DINOv2 ViT-B + synthetic data (recommended)** | Changen2 synthetic + LEVIR-CD | Frozen ViT-B/14 + FPN decoder | Binary CD |
| **DINOv2 ViT-B + land cover, from -> to (recommended)** | as above + OpenEarthMap + SECOND | Same backbone + land-cover head | Labelled CD |
| DINOv2 ViT-B/14 (generalizable) | LEVIR-CD | Frozen ViT-B/14 + FPN decoder | Binary CD |
| DINOv2 ViT-B/14 (fine-tuned) | LEVIR-CD + domain data | Frozen ViT-B/14 + FPN decoder | Binary CD |

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
9. Choose an output GeoPackage (semantic mode also saves before/after land-cover rasters next to it)
10. Click **Run**

### Manual Weight Download

If the installer cannot download weights automatically, download them from the [v0.8.0 release](https://github.com/ja1902/ChangesDetector/releases/tag/v0.8.0) and place them in the project root:

- `dinov2_vitb14_c2s1_levir.pth` (382 MB) -- DINOv2 + synthetic data, the recommended binary model (also finds the changes in semantic mode)
- `landcover_dinov2_vitb14_oem_second.pth` (14 MB) -- land-cover head for labelled (from -> to) change
- `dinov2_vitb14_levir.pth` (382 MB) -- DINOv2 generalizable (v0.6)
- `dinov2_vitb14_egybcd.pth` (382 MB) -- DINOv2 fine-tuned (v0.6)

Model weights are for research / non-commercial use.

### Standalone CLI

```bash
# Binary change detection: recommended DINOv2 model, tested settings, polygons
python detect_changes.py --before before.tif --after after.tif --output-gpkg changes.gpkg

# Labelled change: polygons with from_class / to_class / change / area
python detect_changes.py --before before.tif --after after.tif --mode semantic     --output-gpkg labelled.gpkg --min-area 20

# Another model, e.g. the v0.6 DINOv2 model with a per-scene threshold
python detect_changes.py --before before.tif --after after.tif     --model-type dinov2 --weights dinov2_vitb14_levir.pth --threshold auto --output-gpkg changes.gpkg
```

Options: `--mode binary|semantic`, `--model-type dinov2|ensemble|dinov2_lc`, `--weights path/to/weights.pth`, `--preset synthetic|ensemble`, `--threshold auto|0.3`, `--target-gsd 0.5`, `--min-area 20`, `--histogram-match`, `--tile-size 252`, `--overlap 64`, `--no-coreg`, `--max-shift 50`, `--coreg-window 1024`, `--device auto|cpu|gpu`

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
