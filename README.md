# ChangeDetection - Satellite Image Change Detection

A research project exploring automated change detection between georeferenced satellite images, delivered as a QGIS plugin.

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

### Semantic CD

As I optimise binary CD further (custom dataset), I will be exploring semantic CD in another repo.

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

### Included Model

| Model | Training Dataset | Architecture | Best For |
|-------|-----------------|--------------|----------|
| ChangerEx (R18) | LEVIR-CD | ResNet-18 + FDAF | Building footprint changes |

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
6. Select device: **Auto**, **CPU**, or **GPU**
7. Set processing parameters (tile size, overlap, threshold)
8. Choose an output GeoPackage path
9. Click **Run**

### Manual Weight Download

If the installer cannot download weights automatically, download them from the [GitHub Releases page](https://github.com/ja1902/ChangeDetection/releases) and place in the project root:

- `ChangerEx_r18-512x512_40k_levircd.pth`

### Standalone CLI

For use outside QGIS:

```bash
python detect_changes.py --before path/to/before.tif --after path/to/after.tif
```

Options: `--threshold 0.3`, `--overlap 64`, `--tile-size 256`, `--weights path/to/weights.pth`

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
