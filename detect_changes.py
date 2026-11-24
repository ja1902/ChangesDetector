"""
Run change detection on your own image pair.

Usage:
    python detect_changes.py --before before.tif --after after.tif --output-gpkg changes.gpkg
    python detect_changes.py --before before.tif --after after.tif --mode semantic --output-gpkg labelled.gpkg

Without --weights / --model-type this runs the finetuned DINOv2 model with the
pipeline it was evaluated with, false-alarm filter included (--preset finetuned
--threshold auto); semantic mode adds the land-cover head and writes from -> to
polygons. Every polygon carries "confidence" (mean change probability inside it) and,
with the filter, "filter_score" (its probability of being real). Other models:
    python detect_changes.py --before b.tif --after a.tif --model-type ensemble \
        --weights dinov2_vitb14_c2s1_levir.pth --preset ensemble --threshold auto
    python detect_changes.py --before b.tif --after a.tif --model-type dinov2 \
        --weights dinov2_vitb14_c2s1_levir.pth --preset synthetic --threshold auto
With a preset, "--threshold auto" means the model's tested threshold.
"""

import sys
import os
import json
import argparse
import contextlib
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

import torch

from uchange_qgis_plugin.model_registry import DEFAULT_WEIGHTS, PRESET_WEIGHTS, PRESETS, preset_threshold


_json_progress = False


def _emit(obj):
    if _json_progress:
        print(json.dumps(obj), flush=True)
    else:
        msg = obj.get("message", "")
        if obj["type"] == "log" and msg:
            print(msg)
        elif obj["type"] == "progress":
            pass
        elif obj["type"] == "device":
            print(f"Device: {obj['device']}")
        elif obj["type"] == "error":
            print(f"ERROR: {msg}")
        elif obj["type"] == "result":
            pass


def _log(msg):
    _emit({"type": "log", "message": msg})


def _progress(pct):
    _emit({"type": "progress", "percent": pct})


def read_image(path):
    """Read image as RGB numpy array. Supports PNG/JPG/TIF via PIL, or georeferenced via GDAL."""
    ext = os.path.splitext(path)[1].lower()

    geo_info = None

    if ext in (".tif", ".tiff"):
        try:
            from osgeo import gdal
            from uchange_qgis_plugin.raster_io import bands_to_rgb8, read_band
            gdal.UseExceptions()
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds is not None:
                bands = []
                for i in range(1, min(ds.RasterCount, 3) + 1):
                    bands.append(read_band(ds.GetRasterBand(i)))
                if bands[0].dtype != np.uint8 and max(b.max() for b in bands) > 255:
                    _log(f"  Warning: {bands[0].dtype} image detected, rescaling to 8-bit")
                img = bands_to_rgb8(bands)
                del bands
                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                geo_info = {
                    "geotransform": ds.GetGeoTransform(),
                    "projection": ds.GetProjection(),
                    "width": ds.RasterXSize,
                    "height": ds.RasterYSize,
                }
                ds = None
                return img, geo_info
        except Exception as e:
            # Reading without GDAL loses the georeferencing (no map coordinates,
            # no pixel size, no co-registration): say so instead of failing silently.
            _log(f"  WARNING: GDAL could not read {os.path.basename(path)} ({type(e).__name__}: {e}); "
                 "reading it without georeferencing")

    img = np.array(Image.open(path).convert("RGB"))
    return img, geo_info


def save_geotiff(path, array, geo_info):
    """Save a single-band array as GeoTIFF with georeferencing."""
    from osgeo import gdal
    from uchange_qgis_plugin.raster_io import write_band
    gdal.UseExceptions()
    h, w = array.shape
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, w, h, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(geo_info["geotransform"])
    ds.SetProjection(geo_info["projection"])
    write_band(ds.GetRasterBand(1), array.astype(np.uint8))
    ds.FlushCache()
    ds = None


# comparison.png is a quick look: each panel is shrunk to at most this many
# pixels on its long side, so a large scene does not need a full-size copy.
PREVIEW_SIDE = 2048


def save_comparison(path, panels):
    """Before | after | result side by side; panels are (H, W, 3) or (H, W) uint8."""
    h, w = panels[0].shape[:2]
    f = min(1.0, PREVIEW_SIDE / float(max(h, w)))
    size = (max(1, int(round(w * f))), max(1, int(round(h * f))))
    small = []
    for p in panels:
        if f < 1.0:
            import cv2
            p = cv2.resize(p, size, interpolation=cv2.INTER_AREA)
        small.append(np.repeat(p[:, :, None], 3, axis=2) if p.ndim == 2 else p)
    # Light compression: 3x faster to write than PIL's default, ~10% larger.
    Image.fromarray(np.concatenate(small, axis=1)).save(path, compress_level=1)


def landcover_labels(model, head, device, binary_mask, before_img, after_img, lc_factor,
                     filter_labels=None):
    """Change blobs and their (from, to) land cover: (blob ids, from_cls, to_cls).

    filter_labels (the filter's change ids) joins the pieces of one filtered change."""
    from PIL import Image
    from uchange_qgis_plugin.landcover import label_blobs, predict_landcover
    from uchange_qgis_plugin.raster_io import _smooth_mask
    from uchange_qgis_plugin.resolution import resample_rgb

    h, w = binary_mask.shape
    maps = []
    for i, (name, img) in enumerate((("before", before_img), ("after", after_img))):
        _log(f"Land cover ({name})...")
        lc = predict_landcover(
            model, head, resample_rgb(img, lc_factor), device,
            progress_fn=lambda c, t, i=i: _progress(85 + 5 * i + int(5 * c / t)))
        if lc.shape != (h, w):
            lc = np.array(Image.fromarray(lc).resize((w, h), Image.NEAREST))
        maps.append(lc)
    return label_blobs(_smooth_mask(binary_mask), *maps, filter_labels=filter_labels)


def label_changes(args, model, head, device, binary_mask, before_img, after_img,
                  geo_info, lc_factor, scores=None):
    """Name every change blob: majority land cover before and after.

    Land cover runs at the pixel-size-corrected resolution (without the change
    model's extra upscaling), which is how the land-cover head was evaluated.
    """
    from collections import Counter

    from uchange_qgis_plugin.landcover import LANDCOVER_CLASSES, LANDCOVER_PALETTE
    from uchange_qgis_plugin.raster_io import change_name, polygonize_labelled_changes, save_semantic_geotiff

    blobs, from_cls, to_cls = landcover_labels(model, head, device, binary_mask,
                                               before_img, after_img, lc_factor,
                                               filter_labels=(scores or {}).get("filter_labels"))
    gt, proj = geo_info["geotransform"], geo_info["projection"]
    out = {"mode": "labelled"}
    for tag, cls in (("from", from_cls), ("to", to_cls)):
        path = os.path.join(args.output, f"landcover_{tag}.tif")
        save_semantic_geotiff(cls[blobs], gt, proj, path, LANDCOVER_CLASSES, LANDCOVER_PALETTE)
        out[f"{tag}_path"] = path

    px_area = abs(gt[1] * gt[5])
    areas = np.bincount(blobs.ravel(), minlength=len(from_cls)) * px_area
    kinds = Counter()
    for k in range(1, len(from_cls)):
        kinds[change_name(LANDCOVER_CLASSES[from_cls[k]], LANDCOVER_CLASSES[to_cls[k]])] += areas[k]
    for kind, area in kinds.most_common(6):
        _log(f"  {kind}: {area:,.0f} map units^2")

    if args.output_gpkg:
        _log(f"Polygonizing to {args.output_gpkg}...")
        total, final = polygonize_labelled_changes(
            blobs, from_cls, to_cls, LANDCOVER_CLASSES, gt, proj, args.output_gpkg,
            min_area=args.min_area * px_area,
            style="convex hull" if args.style == "convex" else args.style, **(scores or {}))
        _log(f"Changes written: {final} (of {total}; the rest under the minimum area)")
        out.update(output_path=args.output_gpkg, total_polys=total, final_polys=final)
    return out


def build_parser():
    parser = argparse.ArgumentParser(description="Change detection on custom images")
    parser.add_argument("--before", required=True, help="Path to before image")
    parser.add_argument("--after", required=True, help="Path to after image")
    parser.add_argument("--weights", default=None,
                        help="Path to model weights (default: the recommended DINOv2 model)")
    parser.add_argument("--mode", choices=["binary", "semantic"], default="binary",
                        help="Detection mode: binary (default) or semantic")
    parser.add_argument("--output", default="change_result", help="Output directory")
    parser.add_argument("--threshold", default=None,
                        help="Change threshold: 'auto' or 0.0-1.0 (default: the preset's "
                             "tested threshold, else 0.5)")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="Tile size in pixels (default 256, or the preset's)")
    parser.add_argument("--overlap", type=int, default=None,
                        help="Tile overlap in pixels (default 0, or the preset's)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-backend", choices=["auto", "openvino", "torch"], default="auto",
                        help="On the CPU: run the DINOv2 passes in OpenVINO (auto: when installed; "
                             "about 1.2-1.4x faster, same results to float32 rounding) or PyTorch")
    parser.add_argument("--threads", type=int, default=None, metavar="N",
                        help="CPU threads for the model on the CPU (default: one fewer than the "
                             "CPU has, when PyTorch would otherwise take every one)")
    parser.add_argument("--max-vram", type=int, default=None, metavar="MB",
                        help="Use at most this much GPU memory (MB, plus ~0.3 GB for the "
                             "CUDA runtime itself); for small or shared GPUs")
    parser.add_argument("--grayscale", action="store_true",
                        help="Convert inputs to grayscale (for grayscale-trained models)")
    parser.add_argument("--histogram-match", action="store_true",
                        help="Match each after tile's colours to the before tile. Not recommended: with the "
                             "current models it finds fewer changes (EXPERIMENTS.md)")
    parser.add_argument("--no-coreg", action="store_true",
                        help="Skip co-registration of input images")
    parser.add_argument("--max-shift", type=int, default=50,
                        help="Max co-registration shift in pixels (default: 50)")
    parser.add_argument("--extent", default=None, metavar="XMIN,YMIN,XMAX,YMAX",
                        help="Process only this area, in the before raster's coordinates "
                             "(the after raster is cut to the same area)")
    parser.add_argument("--coreg-window", type=int, default=1024,
                        help="Co-registration matching window size (default: 1024)")
    parser.add_argument("--json-progress", action="store_true",
                        help="Output JSON lines for machine-readable progress")
    parser.add_argument("--output-gpkg", default=None,
                        help="Polygonize binary mask and save as GeoPackage (binary mode only)")
    parser.add_argument("--min-area", type=int, default=0,
                        help="Minimum change area in pixels (used with --output-gpkg)")
    parser.add_argument("--min-area-m2", type=float, default=None,
                        help="Minimum change area in square metres (replaces --min-area; "
                             "needs georeferenced images)")
    parser.add_argument("--style", choices=["exact", "simplified", "convex"],
                        default="exact",
                        help="Polygon simplification style (used with --output-gpkg)")
    parser.add_argument("--model-type", default=None,
                        help="Model type (default dinov2, or dinov2_lc with --mode semantic): dinov2, ensemble "
                             "(DINOv2 weights + Changen2 ChangeStar ViT-L, needs torchange), or "
                             "dinov2_lc (DINOv2 change + land cover: labelled from -> to polygons)")
    parser.add_argument("--preset", default=None, choices=sorted(PRESETS),
                        help="Evaluated inference settings for a model (fills the options below)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Upsample images by this factor before tiling (DINOv2 presets: 1.4)")
    parser.add_argument("--tta", dest="tta", action="store_true", default=None,
                        help="Average predictions over horizontal/vertical flips")
    parser.add_argument("--no-tta", dest="tta", action="store_false",
                        help="No flip averaging, even if the preset uses it")
    parser.add_argument("--tta-gate", type=float, default=None, metavar="P",
                        help="Flip-average only tiles where some pixel's single-pass change "
                             "probability reaches P (faster; tiles with no likely change keep "
                             "the single pass; default: the preset's, 0 = every tile)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: no flip averaging (about 4x faster, slightly less accurate). "
                             "Keeps the preset's upscaling (add --scale 1.0 to drop it too) and uses "
                             "the preset's fast-mode threshold")
    parser.add_argument("--logit-adjust", action="store_true", default=None,
                        help="Remove the LEVIR change prior from the logits (thresholds of the presets assume it)")
    parser.add_argument("--target-gsd", type=float, default=None,
                        help="Resample imagery coarser than this pixel size (m) up to it; 0 = off")
    parser.add_argument("--core-threshold", type=float, default=None,
                        help="Speckle removal: keep a change blob only if it has a pixel at or "
                             "above this probability (0 = off)")
    parser.add_argument("--min-blob-m2", type=float, default=None,
                        help="Speckle removal: drop change blobs smaller than this many m2")
    parser.add_argument("--min-width-m", type=float, default=None,
                        help="Speckle removal: drop blobs whose confident core is thinner than this (m)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Turn the preset's speckle removal off")
    parser.add_argument("--label-free", type=float, default=None, metavar="W",
                        help="Nudge the change probability by the label-free change score "
                             "(uchange_qgis_plugin/labelfree.py) with weight W: "
                             "sigmoid(logit(p) + W * z). Experimental, off by default; not "
                             "used together with the false-alarm filter")
    parser.add_argument("--no-label-free", action="store_true",
                        help="Turn the label-free nudge off")
    parser.add_argument("--landcover-head", default="landcover_dinov2_vitb14_oem_second.pth",
                        help="Land-cover head for --model-type dinov2_lc (labelled change)")
    parser.add_argument("--filter-false-alarms", action="store_true", default=None,
                        help="Drop detected changes a classifier judges false (lean, shadows) and redraw the "
                             "rest at the threshold (uchange_qgis_plugin/falsealarm_filter.py). Needs a "
                             "preset with a false-alarm filter; on by default for those (finetuned)")
    parser.add_argument("--no-filter-false-alarms", dest="filter_false_alarms",
                        action="store_false", help="Turn the false-alarm filter off")
    parser.add_argument("--filter-min-score", type=float, default=None, metavar="P",
                        help="Keep detected changes the filter judges real with at least this probability "
                             "(default: the preset's, 0.46). Lower keeps more")
    parser.add_argument("--no-filter-orphans", action="store_true",
                        help="With the false-alarm filter: do not look for weaker changes (local peaks "
                             "at 0.45 outside the detected ones) for the filter to judge")
    return parser


def resolve_settings(args):
    """Fill model, weights, preset options and fast mode in place; returns the preset."""

    # Defaults: the recommended model, run the way it was tested: finetuned with its
    # false-alarm filter (semantic mode adds the land-cover head).
    if args.weights is None and args.model_type is None and args.preset is None:
        args.preset = "finetuned"
    if args.threshold is None:
        args.threshold = "auto" if args.preset else "0.5"
    if args.model_type is None:
        args.model_type = "dinov2_lc" if args.mode == "semantic" else "dinov2"
    if args.model_type not in ("dinov2", "ensemble", "dinov2_lc"):
        sys.exit(f"Unknown --model-type {args.model_type}: dinov2, ensemble or dinov2_lc "
                 "(ChangerEx and the SCD UPerNet were removed in v0.10.2)")
    if args.mode == "semantic" and args.model_type != "dinov2_lc":
        sys.exit("--mode semantic runs --model-type dinov2_lc (DINOv2 change + land cover)")
    if args.weights is None:
        args.weights = PRESET_WEIGHTS.get(args.preset, DEFAULT_WEIGHTS)

    preset = PRESETS.get(args.preset, {}) if args.preset else {}
    for key, fallback in (("scale", 1.0), ("tta", False), ("logit_adjust", False),
                          ("target_gsd", 0.0), ("tile_size", 256), ("overlap", 0), ("tta_gate", 0.0),
                          ("core_threshold", 0.0), ("min_blob_m2", 0.0), ("min_width_m", 0.0),
                          ("label_free", 0.0)):
        if getattr(args, key) is None:
            setattr(args, key, preset.get(key, fallback))
    if args.fast:
        args.tta = False
    if args.no_cleanup:
        args.core_threshold = args.min_blob_m2 = args.min_width_m = 0.0
    if args.no_label_free:
        args.label_free = 0.0
    bf = preset.get("false_alarm_filter")
    # binary polygons, or labelled polygons (the filter runs on their change map)
    polygon_run = ((args.mode == "binary" and args.model_type == "dinov2")
                   or args.model_type == "dinov2_lc")
    usable = (bf is not None and polygon_run
              and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), bf["file"])))
    if args.filter_false_alarms is None:          # default: on wherever it can run
        args.filter_false_alarms = usable
        if bf is not None and not usable and polygon_run:
            _log(f"False-alarm filter weights not found ({bf['file']}): filter off")
    elif args.filter_false_alarms and not usable:
        sys.exit("--filter-false-alarms needs a binary or labelled run with a preset that has a "
                 "false-alarm filter (--preset finetuned or finetuned_s2l) and its weights file")
    if args.filter_false_alarms:
        args.label_free = 0.0   # the filter was trained on maps without the nudge

    return preset


def resolve_threshold(args, preset):
    """Set args.threshold (float) and args.auto_threshold from --threshold."""
    if args.threshold == "auto" and "threshold" in preset:
        args.auto_threshold = False
        args.threshold = preset_threshold(preset, args.tta, args.label_free)
        fast = ""
        if preset.get("tta") and not args.tta:
            fast += (" (fast mode: no flip averaging)" if "fast_threshold" in preset else
                     " (fast mode: no flip averaging; threshold tested with flip averaging)")
        if not getattr(args, "filter_false_alarms", False):   # with the filter, it sets the outlines
            _log(f"Recommended threshold for this model: {args.threshold:.2f}{fast}")
    elif args.threshold == "auto":
        args.auto_threshold = True
        args.threshold = 0.5  # placeholder, will be computed after inference
    else:
        args.auto_threshold = False
        args.threshold = float(args.threshold)
        if not 0.0 <= args.threshold <= 1.0:
            _log(f"WARNING: threshold {args.threshold} outside [0, 1], clamping")
            args.threshold = max(0.0, min(1.0, args.threshold))
    # The gate is only safe well below the threshold: a pixel just under it in
    # the single pass can still cross the threshold after flip averaging.
    if args.tta_gate and not args.auto_threshold and args.threshold < args.tta_gate + 0.1:
        _log(f"Flip averaging on every tile (threshold {args.threshold:.2f} is close to "
             f"the {args.tta_gate:.2f} tile gate)")
        args.tta_gate = 0.0
    if args.tta_gate and args.auto_threshold:
        args.tta_gate = 0.0


def cpu_threads(requested=None):
    """Threads for the model on the CPU. PyTorch takes every logical CPU where it sees no
    hyperthreading; one busy core (the tile preparation, QGIS, another program) then
    stalls all the others at each step: on a 24-thread machine with one other busy
    process, 23 threads ran a tile in 0.76 s against 1.15 s with 24. Same results."""
    if requested:
        return max(1, requested)
    n, total = torch.get_num_threads(), os.cpu_count() or 1
    return max(1, total - 1) if n >= total and total > 2 else n


def infer_probability(args, model, cs_model, before_img, after_img, factor, device,
                      progress_fn=None):
    """Binary mode: change probability at the input resolution."""
    from uchange_qgis_plugin.resolution import resample_prob, resample_rgb
    from uchange_qgis_plugin.tiling import run_scaled_inference

    tile_kw = dict(tile_size=args.tile_size, overlap=args.overlap, device=device,
                   grayscale=args.grayscale, progress_fn=progress_fn,
                   hist_match=args.histogram_match, tta=args.tta,
                   logit_adjust=args.logit_adjust, tta_gate=args.tta_gate or None)
    # A strip at a time: the upsampled images never sit in memory whole.
    result = run_scaled_inference(model, before_img, after_img, factor, **tile_kw)
    if cs_model is not None:
        # ChangeStar was evaluated at native scale: only the GSD correction applies.
        from uchange_qgis_plugin.changestar_bridge import changestar_probs
        cs_factor = factor / args.scale
        _log("Running Changen2 ChangeStar ViT-L...")
        p_cs = changestar_probs(cs_model, resample_rgb(before_img, cs_factor),
                                resample_rgb(after_img, cs_factor), device)
        p_cs = resample_prob(p_cs, before_img.shape[:2])
        # Geometric mean: a pixel is change only if both models lean that way.
        result = np.sqrt(np.clip(result, 0, 1) * np.clip(p_cs, 0, 1)).astype(np.float32)
    return result


def main():
    global _json_progress

    args = build_parser().parse_args()
    preset = resolve_settings(args)

    _json_progress = args.json_progress

    # Labelled change runs the binary pipeline, then names each change blob.
    labelled = args.model_type == "dinov2_lc"
    if labelled:
        args.mode = "binary"

    threshold_was_auto = args.threshold == "auto"
    resolve_threshold(args, preset)

    for path, label in [(args.before, "Before image"), (args.after, "After image")]:
        if not os.path.isfile(path):
            _emit({"type": "error", "message": f"{label} not found: {path}"})
            sys.exit(1)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "gpu":
        if not torch.cuda.is_available():
            _emit({"type": "error", "message": "CUDA GPU is not available on this system."})
            sys.exit(1)
        device = torch.device("cuda")
    else:
        device = torch.device(args.device)
    _emit({"type": "device", "device": str(device)})
    if device.type == "cpu":
        n = cpu_threads(args.threads)
        torch.set_num_threads(n)
        _log(f"CPU threads: {n}")
    use_ov = False
    if device.type == "cpu" and args.cpu_backend != "torch":
        from uchange_qgis_plugin import openvino_cpu
        ov_version = openvino_cpu.version()
        if ov_version is None and args.cpu_backend == "openvino":
            _emit({"type": "error", "message": "OpenVINO is not installed (pip install openvino)."})
            sys.exit(1)
        use_ov = ov_version is not None
        _log(f"CPU backend: OpenVINO {ov_version}" if use_ov else
             "CPU backend: PyTorch (installing OpenVINO makes CPU runs about 1.2-1.4x faster)")
    if args.max_vram and device.type == "cuda":
        from uchange_qgis_plugin.tiling import set_vram_limit
        set_vram_limit(args.max_vram, device)
        _log(f"GPU memory limit: {args.max_vram} MB")

    from uchange_qgis_plugin.model_bridge import build_model

    project_root = os.path.dirname(os.path.abspath(__file__))
    weights_path = args.weights
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(project_root, weights_path)
    if not os.path.isfile(weights_path):
        _emit({"type": "error", "message": f"Model weights not found: {weights_path}"})
        sys.exit(1)
    model_type = args.model_type

    patch_size = 14
    adjusted = (args.tile_size // patch_size) * patch_size
    if adjusted < patch_size:
        adjusted = patch_size
    if adjusted != args.tile_size:
        _log(f"DINOv2: tile_size {args.tile_size} → {adjusted} (must be multiple of {patch_size})")
        args.tile_size = adjusted

    _log("Building model...")
    model, load_summary = build_model(
        weights_path, device,
        model_type="dinov2")
    _log(load_summary)
    lc_head = None
    if labelled:
        from uchange_qgis_plugin.landcover import load_landcover_head
        head_path = args.landcover_head
        if not os.path.isabs(head_path):
            head_path = os.path.join(project_root, head_path)
        if not os.path.isfile(head_path):
            _emit({"type": "error", "message": f"Land-cover head not found: {head_path}"})
            sys.exit(1)
        lc_head = load_landcover_head(head_path, device)
        _log(f"Land-cover head: {os.path.basename(head_path)}")
    cs_model = None
    if model_type == "ensemble":
        from uchange_qgis_plugin.changestar_bridge import build_changestar
        try:
            cs_model, cs_summary = build_changestar(device)
        except RuntimeError as e:
            _emit({"type": "error", "message": str(e)})
            sys.exit(1)
        _log(cs_summary)
    _progress(20)

    if args.extent:
        import atexit
        import shutil
        import tempfile
        from uchange_qgis_plugin.raster_io import crop_to_extent
        try:
            ext = [float(v) for v in args.extent.split(",")]
            if len(ext) != 4:
                raise ValueError("--extent needs four numbers: XMIN,YMIN,XMAX,YMAX")
            area_dir = tempfile.mkdtemp(prefix="uchange_area_")
            atexit.register(shutil.rmtree, area_dir, True)
            args.before, args.after, used, units = crop_to_extent(args.before, args.after, ext, area_dir)
        except Exception as e:
            _emit({"type": "error", "message": f"Area to process: {e}"})
            sys.exit(1)
        _log(f"Area to process: {used[0]:.1f}, {used[1]:.1f} - {used[2]:.1f}, {used[3]:.1f} "
             f"({used[2] - used[0]:.0f} x {used[3] - used[1]:.0f} {units})")

    _log(f"Reading before: {args.before}")
    before_img, before_geo = read_image(args.before)
    _progress(5)
    _log(f"Reading after:  {args.after}")
    after_img, after_geo = read_image(args.after)
    _progress(10)

    coreg_cleanup = None
    if not args.no_coreg and before_geo and after_geo:
        _log("Co-registering images...")
        try:
            from uchange_qgis_plugin.coregistration import coregister_images
            coreg_result = coregister_images(
                args.before, args.after,
                max_shift=args.max_shift,
                window_size=(args.coreg_window, args.coreg_window),
            )
            if coreg_result.success and coreg_result.corrected_path:
                _log(f"  Shift: X={coreg_result.shift_x_px:.2f}px, Y={coreg_result.shift_y_px:.2f}px")
                after_img, after_geo = read_image(coreg_result.corrected_path)
                coreg_cleanup = coreg_result.cleanup
            elif coreg_result.success:
                _log(f"  {coreg_result.message}")
            else:
                msg = coreg_result.message.rstrip('.')
                _log(f"  {msg}. Using original images.")
        except ImportError as e:
            # Name the missing module: AROSICS itself may be installed while
            # one of its dependencies is not.
            _log(f"  AROSICS not available ({e}). Skipping co-registration.")
        except Exception as e:
            _log(f"  Co-registration failed: {e}. Using original images.")
    _progress(15)

    if before_img.shape[:2] != after_img.shape[:2]:
        bh, bw = before_img.shape[:2]
        ah, aw = after_img.shape[:2]
        if abs(bh - ah) <= 2 and abs(bw - aw) <= 2:
            h_min, w_min = min(bh, ah), min(bw, aw)
            _log(f"  Trimming to common size: {w_min}x{h_min} (was {bw}x{bh} / {aw}x{ah})")
            before_img = before_img[:h_min, :w_min]
            after_img = after_img[:h_min, :w_min]
        else:
            _emit({"type": "error", "message": f"Image dimensions don't match: before={before_img.shape[:2]}, after={after_img.shape[:2]}"})
            sys.exit(1)

    h, w = before_img.shape[:2]
    _log(f"Image size: {w}x{h}")
    if args.min_area_m2:
        from uchange_qgis_plugin.resolution import pixel_size_m
        px_m = pixel_size_m(before_geo or after_geo)
        if px_m:
            args.min_area = args.min_area_m2 / (px_m * px_m)       # in pixels from here on
            _log(f"Minimum change area: {args.min_area_m2:g} m2 ({args.min_area:.0f} pixels)")
        else:
            _log("Minimum change area in m2 ignored: the images are not georeferenced")
    thresh_str = "auto" if args.auto_threshold else f"{args.threshold}"
    if args.filter_false_alarms and threshold_was_auto:
        thresh_str = "set by the false-alarm filter"
    _log(f"Tile size: {args.tile_size}, overlap: {args.overlap}, threshold: {thresh_str}")

    # Resolution: the model's own scale, times an upsampling for coarse imagery.
    from uchange_qgis_plugin.resolution import working_factor
    factor, gsd, gsd_note = working_factor(before_geo or after_geo, args.target_gsd,
                                           args.scale, before_img.shape)
    if gsd:
        _log(f"Pixel size: {gsd:.2f} m")
        km2 = w * h * gsd * gsd / 1e6
        if device.type == "cpu" and km2 >= 0.1:
            # measured on a 24-thread CPU: ~22 min per km2, ~12 with --fast
            _log(f"Running on the CPU: expect roughly {22 * km2:.0f} minutes for {km2:.2f} km2"
                 + ("" if not args.tta else f" (about {12 * km2:.0f} with --fast)")
                 + "; a GPU needs under a minute per km2."
                 + ("" if args.extent else " Try a small area first (--extent)."))
    if gsd_note:
        _log(f"  {gsd_note}")
    if abs(factor - 1.0) > 1e-6:
        _log(f"Inference at x{factor:.2f} ({int(w * factor)}x{int(h * factor)} px)")
    if args.tta or args.logit_adjust:
        tta = "off"
        if args.tta:
            tta = (f"on tiles reaching {args.tta_gate:.2f}" if args.tta_gate else "on")
        _log(f"Flip averaging: {tta}, "
             f"logit adjustment: {'on' if args.logit_adjust else 'off'}")

    os.makedirs(args.output, exist_ok=True)

    t0 = time.time()
    _log("Running tiled inference...")

    def progress_fn(current, total):
        pct = 20 + int(60 * current / total)
        _progress(pct)

    if args.histogram_match:
        _log("Histogram matching: per-tile (after → before)")

    try:
        backend = (openvino_cpu.change_model(model, weights_path, args.tile_size,
                                             torch.get_num_threads(), _log)
                   if use_ov else contextlib.nullcontext())
        with backend:
            result = infer_probability(args, model, cs_model, before_img, after_img, factor,
                                       device, progress_fn)
        del cs_model
        if args.label_free:
            from uchange_qgis_plugin.labelfree import label_free_z, nudge
            t1 = time.time()
            _log(f"Label-free change score (weight {args.label_free:g})...")
            z = label_free_z(model, before_img, after_img, factor, device,
                             progress_fn=lambda i, n: _progress(80 + int(5 * i / n)))
            nudge(result, z, args.label_free)
            del z
            _log(f"  {time.time() - t1:.1f}s")
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and args.max_vram:
            _emit({"type": "error", "message": (
                f"GPU out of memory under --max-vram {args.max_vram}: the DINOv2 models need "
                "at least about 400 MB (weights plus one tile).")})
            sys.exit(1)
        if "out of memory" in str(e).lower():
            _emit({"type": "error", "message": (
                "GPU out of memory. Close other programs using the GPU, or run on the CPU "
                "(--device cpu). The Changen2 ensemble needs about 3.5 GB; the other models "
                "run in under 1 GB.")})
            sys.exit(1)
        raise
    elapsed = time.time() - t0
    _log(f"Inference: {elapsed:.1f}s")

    if not labelled and not args.filter_false_alarms:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    geo_info = before_geo or after_geo
    if not geo_info:
        h, w = before_img.shape[:2]
        geo_info = {
            "geotransform": (0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            "projection": "",
            "width": w,
            "height": h,
        }
    _progress(85)

    prob_map = result
    _log(f"Prob map: min={prob_map.min():.4f}, max={prob_map.max():.4f}, mean={prob_map.mean():.4f}")

    if args.auto_threshold:
        from uchange_qgis_plugin.tiling import auto_threshold
        args.threshold = auto_threshold(prob_map)
        _log(f"Auto threshold: {args.threshold:.6f}")

    from uchange_qgis_plugin.cleanup import clean_mask, to_pixels
    from uchange_qgis_plugin.resolution import pixel_size_m
    area_px, width_px = to_pixels(gsd or pixel_size_m(before_geo or after_geo),
                                  args.min_blob_m2, args.min_width_m)
    if args.filter_false_alarms:
        from uchange_qgis_plugin.falsealarm_filter import filter_mask, load_filter
        bf = preset["false_alarm_filter"]
        flt = load_filter(os.path.join(project_root, bf["file"]))
        # auto: the redraw threshold tested with the filter; a set threshold is used as is
        redraw = bf["redraw"] if threshold_was_auto else args.threshold
        low = min(bf["low"], redraw)
        min_score = bf["min_score"] if args.filter_min_score is None else args.filter_min_score
        orphan_peak = None if args.no_filter_orphans else bf.get("orphan_peak")
        if device.type == "cuda":
            torch.cuda.empty_cache()        # the inference's cached blocks, for the filter
        t1 = time.time()
        _log(f"False-alarm filter: changes at {low:.2f}, keep if judged real >= {min_score:.2f}, "
             f"outlines at {redraw:.2f}...")
        tokens_fn = (openvino_cpu.filter_tokens(model.backbone, weights_path, flt["size"],
                                                flt["layers"], torch.get_num_threads(), _log)
                     if use_ov else None)
        binary_mask, fstats = filter_mask(
            prob_map, before_img, after_img, model.backbone,
            gsd or pixel_size_m(before_geo or after_geo) or 0.5, flt, device,
            low=low, redraw=redraw, min_score=min_score, orphan_peak=orphan_peak,
            progress_fn=lambda i, n: _progress(85 + int(10 * i / n)), tokens_fn=tokens_fn)
        weak = (f", and {fstats['orphans_kept']} of {fstats['orphans']} weaker ones"
                if orphan_peak else "")
        _log(f"  kept {fstats['kept']} of {fstats['blobs']} changes{weak} ({time.time() - t1:.1f}s; "
             f"{fstats['finished']} close enough to the cut to need all four rotations)")
        args.threshold = redraw
        scores = {"prob": prob_map, "filter_labels": fstats["labels"],
                  "filter_real": fstats["real_prob"]}
        if not labelled:                    # land cover still needs the backbone
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        scores = {"prob": prob_map}
        binary_mask = clean_mask(prob_map, args.threshold, args.core_threshold or None,
                                 area_px, width_px)
    if not args.filter_false_alarms and (args.core_threshold or args.min_blob_m2 or args.min_width_m):
        raw = int((prob_map > args.threshold).sum())
        _log(f"Speckle removal (core {args.core_threshold:.2f}, min {args.min_blob_m2:g} m2, "
             f"min width {args.min_width_m:g} m): {raw - int(binary_mask.sum())} pixels removed")
    n_change = int(binary_mask.sum())
    total = binary_mask.size
    _log(f"Change pixels: {n_change}/{total} ({n_change/total:.2%})")

    result_info = {
        "type": "result",
        "mode": "binary",
        "stats": {"change_pixels": n_change, "total_pixels": total},
        "threshold": args.threshold,
    }

    if labelled:
        result_info.update(label_changes(
            args, model, lc_head, device, binary_mask, before_img, after_img,
            geo_info, factor / args.scale, scores))
    elif args.output_gpkg and geo_info:
        _log(f"Polygonizing to {args.output_gpkg}...")
        from uchange_qgis_plugin.raster_io import polygonize_mask

        pixel_w = abs(geo_info["geotransform"][1])
        pixel_h = abs(geo_info["geotransform"][5])
        min_area_map = args.min_area * pixel_w * pixel_h

        style = "convex hull" if args.style == "convex" else args.style
        total_polys, final_polys = polygonize_mask(
            binary_mask, geo_info["geotransform"], geo_info["projection"],
            args.output_gpkg, min_area=min_area_map, style=style, **scores,
        )
        _log(f"Changes written: {final_polys} (of {total_polys}; the rest under the minimum area)")
        result_info["output_path"] = args.output_gpkg
        result_info["total_polys"] = total_polys
        result_info["final_polys"] = final_polys

    elif geo_info:
        mask_path = os.path.join(args.output, "mask.tif")
        save_geotiff(mask_path, binary_mask * 255, geo_info)
        _log(f"Saved: {mask_path} (georeferenced)")
        result_info["output_path"] = mask_path
    else:
        mask_path = os.path.join(args.output, "mask.png")
        Image.fromarray(binary_mask * 255).save(mask_path)
        _log(f"Saved: {mask_path}")
        result_info["output_path"] = mask_path

    comparison_path = os.path.join(args.output, "comparison.png")
    save_comparison(comparison_path, [before_img, after_img, binary_mask * 255])
    _log(f"Saved: {comparison_path}")

    prob_vis = np.empty(prob_map.shape, dtype=np.uint8)
    for y in range(0, prob_map.shape[0], 1024):     # no full-size float temporaries
        prob_vis[y:y + 1024] = (prob_map[y:y + 1024] * 255).clip(0, 255)
    prob_path = os.path.join(args.output, "probability.png")
    Image.fromarray(prob_vis).save(prob_path, compress_level=1)
    _log(f"Saved: {prob_path}")

    _progress(100)
    _emit(result_info)

    if coreg_cleanup:
        coreg_cleanup()

    if not _json_progress:
        print(f"\nDone! Results in {args.output}/")


if __name__ == "__main__":
    main()
