# Changelog

What changed in each version of the Changes Detector plugin, newest first. Experiment
details and every result behind these choices are in [EXPERIMENTS.md](EXPERIMENTS.md).

## v0.10

### Process only an area, one feature per change (v0.10.1)

- **Process only an area**: the dialog has a "Process only an area" box (the current map
  view, a layer's extent, or a rectangle drawn on the map); CLI `--extent XMIN,YMIN,XMAX,YMAX`
  in the before raster's coordinates. Both rasters are cut to that area before anything
  else runs. On a test pair, a 500 m area found every change a whole-image run found there
  (and one more); outlines can differ slightly because the tiles fall in different places.
- **One feature per change**: redrawing a kept change at the filter's stricter outline could
  split it into pieces (on one test pair 22 changes came out as 42 polygons, one site in 13).
  Each change is now a single feature (a MultiPolygon when it has several parts), in binary
  and labelled mode, and `--min-area` applies to the whole change. Pixels and scores are
  unchanged.
- **Without a GPU**: a km² takes about half an hour on a 24-thread CPU (about 18 minutes with
  Fast mode, under a minute on a GPU). The dialog and the CLI now say how long to expect for
  the area chosen. The false-alarm filter judges four changes per pass: same scores, about a
  fifth faster.
- **Minimum change area in m²**: the dialog's minimum area is now in square metres
  (CLI `--min-area-m2`), so it means the same on 0.4 m and 0.5 m imagery; `--min-area` in
  pixels still works.
- **The dialog runs each model with its tested tile settings**: it used to send tile overlap
  32, overriding the tested 64 in every QGIS run; results in QGIS now match the CLI.
- **"Match image histograms" removed from the dialog**: with the current models it finds
  fewer changes (see below); `--histogram-match` still exists.

### The filter also judges weaker changes

Small changes often never reach the model's 0.6 level, so the filter never saw them.
It now also looks at the local peaks of the change map that reach 0.45 outside the
detected changes, judges each the same way, and keeps the ones it trusts as much as
the others. On all 13 areas of the custom dataset this raises the share found at 20 false alarms / km2 from 73% to
80%, mostly changes under 100 m2; at the default "keep if real" it changes little,
so lower that setting to see the gain. `--no-filter-orphans` turns it off.

### An alternative model for oblique imagery

**DINOv2 ViT-B finetuned, S2Looking (oblique imagery)** is the same model finetuned
on the same custom dataset, but alongside S2Looking (off-angle satellite pairs)
instead of LEVIR-CD, with its own false-alarm filter. It may suit oblique imagery
and leaning buildings: on the 13 areas it finds more changes overall (70% against 66%
with the filter, averaged over 5-28 false alarms / km2), with large gains in some areas
and a clear loss in others, so it does not replace the recommended model. On two pairs
from outside the training data it was level with the recommended model on one and slightly
worse on the other (16 against 19 changes found at the default setting). Binary mode
now lists it between the two others. Try it if the recommended model misses changes.
From the command line: `--preset finetuned_s2l`. The installer builds it like the other
models (one more 35 MB decoder in `weights/`).

## v0.9

### New default model: DINOv2 ViT-B finetuned

**DINOv2 ViT-B finetuned** (`dinov2_vitb14_finetuned.pth`) is the recommended
model's decoder

What it brings:

- **More kinds of change.** Besides new buildings it finds construction sites,
  demolitions and cleared land.
- **A false-alarm filter, on by default.** Most false alarms on high-resolution
  imagery are tall buildings leaning differently between the two dates and shadows
  that moved. The filter takes a second look at every detected change: a classifier
  on the frozen DINO features of its before and after crops, trained on 1,711
  hand-checked detections, judges whether it is a real change, and those judged false
  are dropped. Kept changes are outlined at the model's usual threshold.
  (`uchange_qgis_plugin/falsealarm_filter.py`, `weights/falsealarm_filter_finetuned.npz`.)

Results, on 8 areas of the custom dataset, each scored by a model that was trained
without that area (share of real changes found):

| | at 10 false alarms / km2 | at 20 false alarms / km2 |
|---|---|---|
| synthetic data (previous default) | 47% | 58% |
| finetuned | 60% | 71% |
| **finetuned + false-alarm filter (default)** | **69%** | **76%** |

On a new area that neither the model nor the filter had seen (23 changes), the
plugin with its default settings finds 18 changes with 21 false alarms / km2;
without the filter, 19 with 52. The filter removes about 60% of the false alarms
and loses one real change.

In the dialog, **"keep if real"** (default 0.46) sets how sure the filter must be:
lower keeps more changes and more false alarms (0.30: 77% found with 22 false
alarms / km2 on the 8 areas). Untick **"Filter false alarms"** to see everything the
model finds at its threshold (0.85). From the command line: `--preset finetuned`,
`--filter-min-score P`, `--no-filter-false-alarms`.

On the public test sets, which label new buildings only, the finetuned model scores
lower than the previous default, partly because it also flags construction sites
and cleared land. If you only need new buildings, the previous default stays
available as **DINOv2 ViT-B + synthetic data (general)**.

### Labelled change uses the finetuned model and filter

Labelled (from -> to) mode now finds its changes with the finetuned model and the
false-alarm filter, so it gets the same gains as binary mode. The land-cover head
reads only the frozen DINOv2 backbone, so the from/to classes are unchanged; it is the
set of changes that improves. Its polygons are exactly the binary-mode polygons.

Some changes have the same land cover on both dates, mostly construction sites (no
class of their own) and some new flat roofs read as paving: about 1 in 8 of the real
changes on the custom dataset. They are real detections whose type is unclear, so
their `change` field reads e.g. "paved / road (class unchanged)" and QGIS draws them
dashed white as "change, type unclear", not as "became paved / road". On the custom
dataset, 77% of new buildings are named "... -> building".

### A score on every polygon

Every output polygon (binary and labelled) now carries:

- `confidence`: the mean change probability inside it (0-1).
- `filter_score`: with the false-alarm filter, its probability of being a real change
  (kept if at least "keep if real", default 0.46). Sort by it to review the borderline
  ones first. A detection redrawn as several polygons gives them all its score.
- `area`, and `id` (labelled polygons: `id` replaces the old `blob` field).

### Two binary models

Binary mode now lists two models: **DINOv2 ViT-B finetuned** (recommended) and
**DINOv2 ViT-B + synthetic data (general)**. The Changen2 ViT-L ensemble and the two
v0.6 DINOv2 models are no longer in the plugin. The CLI still runs the Changen2
ensemble (`--model-type ensemble --preset ensemble`) for research. The installer no
longer sets up Changen2 by default (`INSTALL_CHANGEN2=1 ./install.sh` adds it) and no
longer builds the v0.6 model files.



## v0.8

### Less GPU memory and RAM, faster (v0.8.3)

Same results as v0.8.2 (on 400 labelled test pairs from S2Looking and LEVIR-CD, scores are identical and 0.001% of pixels change side), measured on a T4:

| | v0.8.2 | v0.8.3 |
|---|---|---|
| Recommended DINOv2 model, 4096 x 4096 px: GPU memory | 4.3 GB | 1.3 GB |
| same, inference time | 1x | 1.6x faster |
| Changen2 ensemble, 2048 x 2048 px: GPU memory | 3.5 GB | 3.0 GB |
| same, inference time | 1x | 1.3x faster |
| 64-megapixel scene: peak RAM | 3.0 GB | 0.9 GB |

- Flip averaging runs only on tiles where the first pass finds a likely change (probability 0.5 or more); the rest keep the single pass. `--tta-gate 0` averages every tile as before.
- The DINOv2 models now run on GPUs with 1 GB of memory: the batch size follows the free memory (8 tiles at most, where the GPU stops getting faster) and halves if the GPU runs out. The Changen2 ensemble needs about 3.5 GB: it now runs in float16, and its attention no longer peaks at ~10 GB.
- **GPU memory limit** in the dialog (CLI: `--max-vram MB`) for small or shared GPUs; same results, a little slower under a tight limit.
- Large rasters are processed one strip at a time, and the comparison preview is capped at 2048 px.

### Labelled change: what changed into what

Semantic mode has a new recommended model, **DINOv2 ViT-B + land cover, from -> to**. The v0.7 DINOv2 model finds the changes; a land-cover head on the same frozen backbone (trained on OpenEarthMap, 44 countries, plus SECOND) classifies each date. Every change polygon gets one before class and one after class (the majority inside it) and is written to a GeoPackage with the fields `from_class`, `to_class`, `change` (e.g. "farmland -> building") and `area`. Classes: bare ground, grass / low vegetation, paved / road, trees, water, farmland, building. The layer is coloured by what each area became.

Share of truly changed pixels whose before *and* after class are right:

| | SECOND (China) | HRSCD (France, never seen) |
|---|---|---|
| SCD UPerNet (v0.7) | 60% | 40% |
| **DINOv2 + land cover (v0.8)** | **63%** | **50%** |

Changes are found by the building-focused v0.7 model, so this mode suits building and urban change: on the French data it found changes far better than the UPerNet (IoU 0.22 vs 0.05). Changes that involve no building, such as forest cleared for farmland, are largely missed. Needs `landcover_dinov2_vitb14_oem_second.pth` (the installer sets it up). CLI: `--model-type dinov2_lc --preset synthetic --threshold auto --output-gpkg out.gpkg`.

### Fast mode

A **Fast mode** checkbox (CLI: `--fast`) for the recommended binary and labelled-change models skips flip averaging: about 2x faster, slightly less accurate. It keeps the 1.4x upscaling (`--scale 1.0` drops that too; `--no-tta` / `--tta` set flip averaging directly).

### Models dropped

ChangerEx (R18) and the SCD UPerNet (R18) are no longer installed or listed: the DINOv2 models beat ChangerEx on every unseen dataset tested, and the new labelled-change mode replaces the SCD UPerNet for building and urban change. Their weights stay on the [v0.6.0 release](https://github.com/ja1902/ChangesDetector/releases/tag/v0.6.0), and the CLI still runs them with `--model-type opencd` / `--model-type opencd_scd --mode semantic`.

### Fixes

- **Weights from the official source**: the DINOv2 models only train a decoder on Meta's frozen DINOv2 ViT-B/14, so the installer now downloads that backbone once (346 MB) from Meta's official release and combines it with the trained decoders, which ship in this repository's `weights/` folder. This replaces three 382 MB downloads from the GitHub release; the resulting files are identical.
- **CPU**: DINOv2 models crashed on computers without an NVIDIA GPU (float16 weights, float32 inputs); they now run in float32 on CPU.
- **Installer**: a failed weight download no longer stops the installer. It now checks GDAL's NumPy support after building the bindings, and installs Changen2 support and its weights by default (`INSTALL_CHANGEN2=0 ./install.sh` skips it).
- **GeoTIFFs on some Linux installs**: when GDAL's Python bindings were built without NumPy support (`No module named '_gdal_array'`), GeoTIFFs were silently read without georeferencing and writing polygons failed. Rasters are now read and written without that module, and a failed GDAL read is reported instead of hidden.
- **CLI defaults**: with no model given, `detect_changes.py` now runs the recommended DINOv2 model with its tested settings (it used to default to ChangerEx).

## v0.7

v0.7 targets three problems: working on new regions, sensors and resolutions without retraining; false alarms on angled (off-nadir) imagery; and naming what changed into what.

### New default model: DINOv2 + synthetic change data

Same architecture and file format as the v0.6 DINOv2 model, but the decoder is trained on [Changen2](https://github.com/Z-Zheng/pytorch-change-models) synthetic change pairs plus LEVIR-CD (`dinov2_vitb14_c2s1_levir.pth`). IoU on test sets, with each model's threshold picked on a *different* unseen dataset (the fair setting for new imagery; v0.6's own auto-threshold scored lower, e.g. 0.20 on S2Looking):

| IoU | S2Looking | LEVIR-CD |
|---|---|---|
| v0.6 DINOv2 (generalizable) | 0.27 | 0.84 |
| **v0.7 DINOv2 + synthetic** | **0.35** | 0.76-0.80 |
| v0.7 Changen2 ViT-L + DINOv2 (optional) | **0.40** | 0.80 |

At the same detection rate it raises about 4x fewer false alarms than v0.6 on imagery of the same place seen from different angles (SpaceNet MVOI). Trained on CC BY-NC-SA 4.0 data: research / non-commercial use.

### Each model runs the way it was tested

Models now carry a preset: 1.4x upscaling, flip averaging, logit adjustment and a threshold chosen on imagery the model had not seen. The "Change threshold" box is now **Recommended**: for preset models it uses that tested threshold, not the per-scene estimate (which reached only ~70% of the best possible IoU on unfamiliar data). CLI: `--preset synthetic` or `--preset ensemble` with `--threshold auto`.

### Coarse imagery is resampled automatically

Every model collapses on imagery much coarser than it was trained on (2 m LEVIR: IoU about 0.1). Rasters whose pixel size is coarser than 0.5 m are now resampled to 0.5 m before detection, and results are written at the original resolution. On 2 m test GeoTIFFs this took the new model from 0.39 to 0.73 IoU. CLI: `--target-gsd 0.5` (0 to turn off).

### Optional: Changen2 ViT-L + DINOv2 (most accurate)

Runs the Changen2 ChangeStar ViT-L model alongside the DINOv2 model and keeps only change both agree on. The installer sets up the `torchange` package and downloads the weights from huggingface.co by default (`INSTALL_CHANGEN2=0` skips this, and the plugin then prints install instructions). It is slower; ChangeStar weights are CC BY-NC-SA 4.0.

### Fixes

- **Semantic change legend**: the SCD UPerNet's outputs were shown with the wrong class names (low vegetation as "water", tree as "low vegetation", water as "tree"). With the corrected order its SeK on SECOND rises from 0.13 to 0.20.
- **Newer GPUs**: DINOv2 models failed on RTX 30-series and newer (float16 weights under bfloat16 autocast); autocast now follows the model's precision.

## v0.6

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

- Optional **per-tile histogram matching** (`--histogram-match`): each "after" tile's colours are
  matched to the "before" tile
- **Not recommended with the current models** and no longer in the dialog (v0.10.1): on the 13
  areas of the training data it found fewer changes (-2.7 points with the false-alarm filter,
  -7.3 without), small ones included, because matching a tile partly erases the change in it

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

## v0.5

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

## v0.4

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

## v0.3

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


## v0.2

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
