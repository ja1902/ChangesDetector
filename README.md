# Changes Detector

A QGIS plugin that finds what changed between two georeferenced satellite or aerial images
of the same place: new buildings, construction sites, demolitions and cleared land. It
outlines every change as a polygon, with a score for how likely it is to be real, and can
also say what each area changed from and into ("farmland -> building").

It runs on a GPU (under a minute per km² at 0.5 m) or on a CPU (about half an hour per km²),
and inference runs in its own Python environment, so it works with any QGIS 3.22+.

- What changed in each version: [CHANGELOG.md](CHANGELOG.md)
- Every experiment and result behind the current models: [EXPERIMENTS.md](EXPERIMENTS.md)

## What to expect

The recommended model is DINOv2 (Meta's ViT-B/14, frozen) with a decoder finetuned on 13
hand-checked areas of 1 km² of high-resolution imagery (0.4-0.5 m), followed by a
**false-alarm filter**: a classifier, trained on 1,711 hand-checked detections, that takes
a second look at every detected change and drops those that look like buildings leaning
differently between the two dates or shadows that moved, the main false alarms at this
resolution. It also judges weaker changes the model alone would miss.

On the 13 areas, each scored by a model trained without that area, the default setup finds
about 80% of the real changes at 20 false alarms per km². Large changes are found far more
often than small ones: changes under 100 m² are the hardest. Results on your own imagery
depend on how much it looks like the training data; the score on each polygon lets you
review the borderline ones first.

## Installation

Requirements:

- **Python 3.10-3.14** (Ubuntu 22.04-26.04 all supported)
- **QGIS 3.22+** (any Python version: the plugin runs inference in a subprocess)
- **NVIDIA GPU** recommended; CPU also works, more slowly
- **System packages** (Linux): `sudo apt install python3-venv python3-dev libgdal-dev build-essential`

Clone the repository and run the installer:

```bash
git clone https://github.com/ja1902/ChangesDetector.git
cd ChangesDetector
./install.sh        # Linux
install.bat         # Windows
```

The installer creates a Python virtual environment, installs PyTorch (with CUDA if a GPU is
found) and the other dependencies, builds the model weights (Meta's DINOv2 backbone plus
the trained parts in `weights/`) and links the plugin into your QGIS plugins folder.

## Using it in QGIS

1. Open QGIS, enable **ChangeDetection** in **Plugins > Manage and Install Plugins**, then
   open it from **Plugins > ChangeDetection**.
2. Choose the **before** and **after** raster layers (georeferenced, e.g. GeoTIFF).
3. Optional: tick **Process only an area** and click **Draw area on map** to drag a
   rectangle (or use the map view or a layer's extent). Much faster on large images or
   without a GPU, and a good way to try the plugin on a small part first.
4. Choose the mode: **Binary Change Detection** (where something changed) or **Semantic
   Change Detection** (also what it changed from and into).
5. Keep the recommended model and settings, or adjust:
   - **Filter false alarms** (on) and **keep if real ≥** (0.46): lower keeps more changes
     and more false alarms;
   - **Fast mode**: about twice as fast on the model step, slightly less accurate;
   - **Min change area** in m²;
   - **Device** (Auto, CPU, GPU) and a **GPU memory limit** for small or shared GPUs.
6. Choose the output GeoPackage and click **Run**. The result is added to the project.

Images are co-registered first (a shift of up to 50 px between the two dates is corrected),
and imagery coarser than 0.5 m is resampled to 0.5 m and the result written back at the
original resolution.

### Many pairs: the Processing toolbox

The plugin also adds **ChangeDetection > Detect changes** to the Processing toolbox, with
the same options (model, area, filter, minimum area, Fast mode, device). Right-click it and
choose **Execute as Batch Process** to run many image pairs in one go, or use it in the
graphical modeler. Results get the same styling as from the dialog.

### Output

One feature per change, with:

| field | meaning |
|---|---|
| `id` | change number |
| `change` | 1 (binary), or e.g. "farmland -> building" (semantic) |
| `from_class`, `to_class` | land cover before and after (semantic only) |
| `confidence` | mean change probability inside the change (0-1) |
| `filter_score` | the false-alarm filter's probability that the change is real; sort by it to review the borderline ones first |
| `area` | area in map units² |

Binary results are outlined by `filter_score`: red (likely real, 0.70 and up), orange
(0.50-0.70), dashed yellow (under 0.50, when "keep if real" is lowered). A change can have
several parts (a MultiPolygon). In semantic mode, changes whose land
cover is the same on both dates (mostly construction sites) read e.g. "paved / road (class
unchanged)" and are drawn as "change, type unclear". Semantic mode also saves before and
after land-cover rasters next to the GeoPackage.

## Models

| Model | Trained on | Mode |
|---|---|---|
| **DINOv2 ViT-B finetuned (recommended)** | Changen2 synthetic + LEVIR-CD + custom dataset, false-alarm filter | Binary |
| DINOv2 ViT-B finetuned, S2Looking (oblique imagery) | as above, with S2Looking in place of LEVIR-CD, its own filter | Binary |
| DINOv2 ViT-B + synthetic data (general) | Changen2 synthetic + LEVIR-CD (new buildings only) | Binary |
| **DINOv2 ViT-B + land cover, from -> to (recommended)** | the finetuned model and filter, land cover from OpenEarthMap + SECOND | Semantic |

The S2Looking variant found more changes than the recommended model on some areas of the
training data and fewer on others; on two areas outside it, it was level on one and slightly
worse on the other. Try it if the recommended model misses changes on angled imagery. The
general model finds new buildings only and scores best on public building datasets.

### Building the weights by hand

The installer does this. If its download of the backbone fails, fetch
[dinov2_vitb14_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)
(346 MB) into the project folder and run:

```bash
python weights/assemble.py . dinov2_vitb14_pretrain.pth
```

This builds `dinov2_vitb14_finetuned.pth`, `dinov2_vitb14_finetuned_s2l.pth`,
`dinov2_vitb14_c2s1_levir.pth` and copies `landcover_dinov2_vitb14_oem_second.pth`. The
backbone file can be deleted afterwards.

## Command line

`detect_changes.py` runs the same pipeline as the plugin (use the virtual environment's
Python, `venv/bin/python`):

```bash
# Binary change: the recommended model with its false-alarm filter and tested settings
python detect_changes.py --before before.tif --after after.tif --output-gpkg changes.gpkg

# Semantic change: from_class / to_class / change on every polygon
python detect_changes.py --before before.tif --after after.tif --mode semantic --output-gpkg labelled.gpkg

# Only part of the image (coordinates of the before raster)
python detect_changes.py --before before.tif --after after.tif --extent 500000,4000000,500500,4000500 --output-gpkg area.gpkg

# The S2Looking variant, or the general model
python detect_changes.py --before before.tif --after after.tif --preset finetuned_s2l --output-gpkg changes.gpkg
python detect_changes.py --before before.tif --after after.tif --weights dinov2_vitb14_c2s1_levir.pth --preset synthetic --output-gpkg changes.gpkg
```

Main options:

| option | |
|---|---|
| `--extent XMIN,YMIN,XMAX,YMAX` | process only this area |
| `--filter-min-score 0.46` | "keep if real" for the false-alarm filter |
| `--no-filter-false-alarms`, `--no-filter-orphans` | turn the filter, or its judging of weaker changes, off |
| `--min-area-m2 50` | drop smaller changes (`--min-area` in pixels also works) |
| `--fast` | skip flip averaging |
| `--device auto\|cpu\|gpu`, `--max-vram MB` | where to run, and a GPU memory limit |
| `--no-coreg`, `--max-shift 50` | co-registration |
| `--target-gsd 0.5` | resample coarser imagery to this pixel size (0: off) |
| `--threshold auto\|0.3` | change threshold (auto: the one each model was tested with) |

`python detect_changes.py --help` lists the rest (tile size and overlap, scale, styles, and
older models kept for research).

## Training on your own imagery

`finetune_dinov2.py` finetunes the decoder on your own labelled areas, mixed with a public
dataset so the model does not forget what it knew; this is how the recommended model was
made. Label your areas with `benchmarks/make_labels.py` (reviewed candidates become
`train_label.tif`), then:

```bash
python finetune_dinov2.py --squares path/to/your_areas --levir path/to/LEVIR-CD --seed 1
```

The script's header documents the options (S2Looking as partner, hard negatives, seeds,
weight averaging). `train_dinov2_cd.py` trains a decoder from scratch on a dataset in
LEVIR-CD format (`train/A/`, `train/B/`, `train/label/`). The backbone stays frozen in
both; about 4 GB of GPU memory is enough.

## Troubleshooting

**Check an installation:** `venv/bin/python tests/smoke_test.py` runs the pipeline on a
small synthetic pair and checks the output (about half a minute on a GPU; `--quick` skips
the full runs).

**The plugin doesn't appear in QGIS:** check it is enabled in the Plugin Manager, that the
link in your QGIS plugins folder exists, and restart QGIS after installing.

**"Model weights not found":** run the installer, or build the weights by hand (above), or
tick "Use custom weights file" to point to your own checkpoint.

**"CUDA GPU is not available":** choose CPU as the device, or install the NVIDIA drivers.

**GPU out of memory:** close other programs that use the GPU. The DINOv2 models need about
0.4 GB (plus about 0.3 GB for the driver); the batch size follows the free memory, and a
"GPU memory limit" below what the model needs fails with a clear message.

**Slow:** use a GPU if you have one; otherwise tick **Process only an area** to run on part
of the image, and **Fast mode** (about 18 instead of 30 minutes per km² on a 24-thread CPU).

**"Process only an area" needs georeferenced rasters:** the area is given in map
coordinates, so images without georeferencing (e.g. plain JPEGs) can only be processed whole.
