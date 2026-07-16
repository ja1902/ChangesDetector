# ChangeDetection - Satellite Image Change Detection

A research project exploring automated change detection between georeferenced satellite images, delivered as a QGIS plugin.

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

### Included Models

| Model | Training Dataset | Architecture | Mode |
|-------|-----------------|--------------|------|
| ChangerEx (R18) | LEVIR-CD | ResNet-18 + FDAF | Binary CD |
| SCD UPerNet (R18) | SECOND | UPerNet + ResNet-18 | Semantic CD |

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

## Roadmap

### Phase 1 (current) -- Pretrained inference plugin

An end-to-end QGIS plugin with a pretrained ChangerEx (R18) checkpoint trained on LEVIR-CD. Clone, install, run. Useful for exploratory analysis and as a baseline before fine-tuning.

### Phase 2 -- Custom dataset creation and fine-tuning

Explore workflows for creating labelled change detection datasets from one's own satellite imagery, and fine-tuning on them. The goal is to close the domain gap for a specific area of interest.

### Phase 3 -- Vision-language models for change detection

Integrate VLMs so that users can either specify the type of change they are looking for (e.g. "new buildings", "deforestation", "road construction") or receive a textual description of what changed between the two images. This would move beyond binary change masks toward semantic, interpretable change analysis.

---

## Phase 1: QGIS Plugin

### Included Models

| Model | Training Dataset | Architecture | Mode |
|-------|-----------------|--------------|------|
| ChangerEx (R18) | LEVIR-CD | ResNet-18 + FDAF | Binary CD |
| SCD UPerNet (R18) | SECOND | UPerNet + ResNet-18 | Semantic CD |

### Prerequisites

- **Python 3.10+**
- **QGIS 3.22+**
- **NVIDIA GPU** (recommended, CPU also supported)

### Installation

Clone the repository and run the installer:

```bash
git clone https://github.com/ja1902/ChangeDetection.git
cd ChangeDetection
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

If the installer cannot download weights automatically, download them from the [GitHub Releases page](https://github.com/ja1902/ChangeDetection/releases) and place in the project root:

- `ChangerEx_r18-512x512_40k_levircd.pth` (binary CD)
- `scd_upernet_r18_10k_second.pth` (semantic CD)

### Standalone CLI

For use outside QGIS:

```bash
python detect_changes.py --before path/to/before.tif --after path/to/after.tif
```

For semantic change detection:
```bash
python detect_changes.py --before path/to/before.tif --after path/to/after.tif --mode semantic
```

Options: `--mode binary|semantic`, `--threshold 0.3`, `--overlap 64`, `--tile-size 256`, `--weights path/to/weights.pth`, `--no-coreg`, `--max-shift 50`, `--coreg-window 1024`

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

**Slow inference:**
- Use GPU if available
- Increase tile size (256 -> 512) to process fewer tiles
- Reduce overlap (64 -> 0) for faster but slightly less smooth results
