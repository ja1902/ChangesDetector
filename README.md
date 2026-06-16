# ChangesDetector - Satellite Image Change Detection

A research project exploring automated change detection between georeferenced satellite images, delivered as a QGIS plugin.

## Research Context

This project investigates deep learning approaches for satellite image change detection. It compares CNNs (U-Net variants, Siamese networks), transformer-based models, and state space models. Two architectures stood out: **MambaBCD** from the [ChangeMamba](https://github.com/ChenHongruixuan/ChangeMamba) framework and **PeftCD** ([DINO3CD](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba), DINOv3 ViT-L backbone with LoRA parameter-efficient fine-tuning).

PeftCD generalised noticeably better to unseen imagery in evaluation. The reason is its backbone: DINOv3 is a Vision Foundation Model pre-trained on a massive and diverse dataset, giving it rich, general-purpose visual representations that transfer across sensors, geographies, and imaging conditions. LoRA fine-tuning then adapts those representations to the change detection task without discarding that breadth. MambaBCD is trained from scratch on change detection datasets alone, so its representations are narrowly tuned to its training distribution.

That said, PeftCD does not fully close the domain gap -- it still struggles on imagery that differs significantly from its training data. Mamba's selective state space design has linear complexity with respect to sequence length, which should be a speed advantage at scale, but in practice the two architectures performed similarly in both speed and output quality on representative test images. The included DINO3CD checkpoints were trained on LEVIR-CD+, SYSU-CD, and a combination of both, and they outperform the publicly available MambaBCD pretrained checkpoints on benchmark data.

### Findings on generalization

A central finding is that most change detection models are **domain-locked** -- they perform well on imagery similar to their training set but struggle on anything else. Several pretrained MambaBCD checkpoints were evaluated across datasets:

- **LEVIR-CD+** (0.5m, Google Earth, building changes) -- strong on building detection in similar imagery, blind to vegetation changes, failed on other satellite datasets
- **SYSU-CD** (0.5m, multi-type changes including vegetation, roads, construction) -- detected vegetation loss that LEVIR-CD+ missed entirely, but produced false positives on buildings
- **WHU-CD** (0.075m, aerial, building changes) -- only detected a fraction of building changes on other satellite datasets, likely due to the resolution mismatch
- **Cross-dataset training (MambaBCD)** -- training on a combination of SYSU-CD and LEVIR-CD+ still caused significant accuracy drops on unseen imagery due to domain gap


#### Training the DINO3CD models

To test whether a Vision Foundation Model backbone could improve generalisation, DINO3CD was trained using PyTorch Lightning on a single GPU. Results:

| Training data | Crop size | Steps | Val IoU | Test IoU | Test F1 |
|--------------|-----------|-------|---------|----------|---------|
| SYSU-CD | 256 | 5,000 | 71.96% | 74.0% | 83.0% |
| LEVIR-CD+ | 256 | — | — | 74.7% | 85.5% |
| LEVIR-CD+ | 512 | 14,000 | 84.03% | 76.5% | **86.7%** |
Key takeaways:
- **DINO3CD outperforms MambaBCD**: MambaBCD achieves ~83% F1 on LEVIR-CD+; the DINO3CD 512-crop model reaches 86.7%
- **Larger crops help**: 512-crop training gives the model more spatial context, gaining ~1.2% F1 over the 256-crop equivalent
- **Best for real-world use**: the SYSU-CD model generalises best to Google Earth imagery -- it detects diverse changes (buildings, vegetation, roads) with few misses. The LEVIR model misses non-building changes entirely since it was only trained on buildings

Even with a VFM backbone, the domain gap is not fully closed -- all models still struggle on imagery that differs significantly from their training data. **The only reliable path to accurate results on a specific area is fine-tuning on labelled data from that region using the same image source.** Phase 2 will explore fine-tuning workflows for both DINO3CD and MambaBCD on custom AOI data.

## Roadmap

### Phase 1 (current) -- Pretrained inference plugin

An end-to-end QGIS plugin with pretrained DINO3CD and MambaBCD checkpoints (LEVIR-CD+ and SYSU-CD). Clone, install, run. The included DINO3CD models generalise better than MambaBCD to unseen imagery, though accuracy on out-of-distribution data is still limited by domain gap. Useful for exploratory analysis and as a baseline before fine-tuning.

### Phase 2 -- Custom dataset creation and fine-tuning

Explore workflows for creating labelled change detection datasets from one's own satellite imagery, and fine-tuning DINO3CD and MambaBCD on them. The goal is to close the domain gap for a specific area of interest. Fine-tuning with even a small amount of local labelled data should significantly improve detection accuracy for both architectures.

### Phase 3 -- Vision-language models for change detection

Integrate VLMs so that users can either specify the type of change they are looking for (e.g. "new buildings", "deforestation", "road construction") or receive a textual description of what changed between the two images. This would move beyond binary change masks toward semantic, interpretable change analysis.

---

## Phase 1: QGIS Plugin

### Included Models

| Model | Training Dataset | Architecture | Test F1 | Best For |
|-------|-----------------|--------------|---------|----------|
| MambaBCD-Small | LEVIR-CD+ | VSSM Small (dims=96) | 88.3% | Building footprint changes |
| MambaBCD-Small | SYSU-CD | VSSM Small (dims=96) | 83.4% | Vegetation, roads, construction, general land-use changes |
| DINO3CD (DINOv3+LoRA) | LEVIR-CD+ (512-crop) | ViT-L + LoRA | **86.7%** | Building footprint changes, better generalisation |
| DINO3CD (DINOv3+LoRA) | SYSU-CD | ViT-L + LoRA | 83.0% | Vegetation, roads, construction, best real-world generalisation |

DINO3CD uses DINOv3 (a Vision Foundation Model pre-trained on large-scale diverse imagery) as its frozen backbone, with LoRA adapters trained for change detection. This gives it broader generalisation to unseen imagery compared to MambaBCD, which is trained from scratch on change detection data alone.

### Prerequisites

- **Python 3.10+**
- **QGIS 3.22+**
- **NVIDIA GPU** (recommended, CPU also supported)

### Installation

Clone the repository and run the installer:

```bash
git clone https://github.com/ja1902/ChangesDetector.git
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
4. Download both model weights (~432 MB total)
5. Link the plugin into your QGIS plugins directory

### Usage

1. Open QGIS
2. Go to **Plugins > Manage and Install Plugins**
3. Enable **"ChangeDetection"**
4. Open the plugin from **Plugins > ChangeDetection**
5. Select your **before** and **after** raster layers
6. Choose a **model**:
   - **LEVIR-CD+** for building footprint changes
   - **SYSU** for vegetation, roads, and general land-use changes
7. Select device: **Auto**, **CPU**, or **GPU**
8. Set processing parameters (tile size, overlap, threshold)
9. Choose an output GeoPackage path
10. Click **Run**

### Manual Weight Download

If the installer cannot download weights automatically, download them from the [GitHub Releases page](https://github.com/ja1902/ChangesDetector/releases/tag/v0.1.0) and place in the project root:

- `MambaBCD_Small_LEVIRCD+.pth` (207 MB)
- `MambaBCD_Small_SYSU.pth` (207 MB)
- `PeftCD_LEVIRCD.ckpt`
- `PeftCD_SYSU.ckpt`

### Standalone CLI

For use outside QGIS:

```bash
python detect_changes.py --before path/to/before.tif --after path/to/after.tif
```

Options: `--threshold 0.3`, `--overlap 64`, `--tile-size 256`, `--weights path/to/weights.pth`

### How the virtual environment works with QGIS

QGIS has its own built-in Python interpreter - when you run a plugin, it always uses that interpreter. The virtual environment created by the installer does **not** replace QGIS's Python; it is only used as a **package library**.

When the plugin needs to import a heavy dependency like PyTorch or transformers, it adds the venv's `site-packages` folder to Python's search path (`sys.path`) at runtime. This makes QGIS's Python find and load packages from the venv as if they were installed normally.

**Why not just pip install into QGIS's Python?** QGIS ships a minimal Python environment. Installing lots of packages directly into it risks version conflicts and makes uninstalling the plugin messy. A separate venv keeps everything isolated - remove the `venv/` folder and the plugin's dependencies are gone.

**In short:** QGIS's Python runs the code, the venv stores the packages, and `sys.path` connects the two.

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
