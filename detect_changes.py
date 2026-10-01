"""
Run change detection on your own image pair.

Usage:
    python detect_changes.py --before before.tif --after after.tif --output-gpkg changes.gpkg
    python detect_changes.py --before before.tif --after after.tif --mode semantic --output-gpkg labelled.gpkg

Without --weights / --model-type this runs the recommended DINOv2 model with the
pipeline it was evaluated with (--preset synthetic --threshold auto); semantic
mode adds the land-cover head and writes from -> to polygons. Other models:
    python detect_changes.py --before b.tif --after a.tif --model-type ensemble \
        --weights dinov2_vitb14_c2s1_levir.pth --preset ensemble --threshold auto
    python detect_changes.py --before b.tif --after a.tif --model-type dinov2 \
        --weights dinov2_vitb14_levir.pth --threshold auto
With a preset, "--threshold auto" means the model's tested threshold.
"""

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

import torch

from uchange_qgis_plugin.model_registry import DEFAULT_WEIGHTS, PRESETS, preset_threshold


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
            from uchange_qgis_plugin.raster_io import read_band
            gdal.UseExceptions()
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds is not None:
                bands = []
                for i in range(1, min(ds.RasterCount, 3) + 1):
                    bands.append(read_band(ds.GetRasterBand(i)))
                img = np.stack(bands, axis=-1)
                if img.dtype != np.uint8:
                    max_val = img.max()
                    if max_val > 255:
                        _log(f"  Warning: {img.dtype} image detected, rescaling to 8-bit")
                        img = (img.astype(np.float64) / max_val * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
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


def label_changes(args, model, head, device, binary_mask, before_img, after_img,
                  geo_info, lc_factor):
    """Name every change blob: majority land cover before and after.

    Land cover runs at the pixel-size-corrected resolution (without the change
    model's extra upscaling), which is how the land-cover head was evaluated.
    """
    from collections import Counter

    from PIL import Image
    from uchange_qgis_plugin.landcover import (
        LANDCOVER_CLASSES, LANDCOVER_PALETTE, label_blobs, predict_landcover)
    from uchange_qgis_plugin.raster_io import (
        _smooth_mask, polygonize_labelled_changes, save_semantic_geotiff)
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

    blobs, from_cls, to_cls = label_blobs(_smooth_mask(binary_mask), *maps)
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
        kinds[f"{LANDCOVER_CLASSES[from_cls[k]]} -> {LANDCOVER_CLASSES[to_cls[k]]}"] += areas[k]
    for kind, area in kinds.most_common(6):
        _log(f"  {kind}: {area:,.0f} map units^2")

    if args.output_gpkg:
        _log(f"Polygonizing to {args.output_gpkg}...")
        total, final = polygonize_labelled_changes(
            blobs, from_cls, to_cls, LANDCOVER_CLASSES, gt, proj, args.output_gpkg,
            min_area=args.min_area * px_area,
            style="convex hull" if args.style == "convex" else args.style)
        _log(f"Polygons: {total} created, {final} after min-area filter")
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
    parser.add_argument("--grayscale", action="store_true",
                        help="Convert inputs to grayscale (for grayscale-trained models)")
    parser.add_argument("--histogram-match", action="store_true",
                        help="Match after image histogram to before image (fixes radiometric mismatch)")
    parser.add_argument("--no-coreg", action="store_true",
                        help="Skip co-registration of input images")
    parser.add_argument("--max-shift", type=int, default=50,
                        help="Max co-registration shift in pixels (default: 50)")
    parser.add_argument("--coreg-window", type=int, default=1024,
                        help="Co-registration matching window size (default: 1024)")
    parser.add_argument("--json-progress", action="store_true",
                        help="Output JSON lines for machine-readable progress")
    parser.add_argument("--output-gpkg", default=None,
                        help="Polygonize binary mask and save as GeoPackage (binary mode only)")
    parser.add_argument("--min-area", type=int, default=0,
                        help="Minimum polygon area in pixels (used with --output-gpkg)")
    parser.add_argument("--style", choices=["exact", "simplified", "convex"],
                        default="exact",
                        help="Polygon simplification style (used with --output-gpkg)")
    parser.add_argument("--model-type", default=None,
                        help="Model type (default dinov2, or dinov2_lc with --mode semantic): dinov2, ensemble "
                             "(DINOv2 weights + Changen2 ChangeStar ViT-L, needs torchange), or "
                             "dinov2_lc (DINOv2 change + land cover: labelled from -> to polygons); "
                             "legacy: opencd (ChangerEx), opencd_scd (SCD UPerNet)")
    parser.add_argument("--preset", default=None, choices=sorted(PRESETS),
                        help="Evaluated inference settings for a model (fills the options below)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Upsample images by this factor before tiling (DINOv2 presets: 1.4)")
    parser.add_argument("--tta", dest="tta", action="store_true", default=None,
                        help="Average predictions over horizontal/vertical flips")
    parser.add_argument("--no-tta", dest="tta", action="store_false",
                        help="No flip averaging, even if the preset uses it")
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
    parser.add_argument("--landcover-head", default="landcover_dinov2_vitb14_oem_second.pth",
                        help="Land-cover head for --model-type dinov2_lc (labelled change)")
    return parser


def resolve_settings(args):
    """Fill model, weights, preset options and fast mode in place; returns the preset."""

    # Defaults: the recommended DINOv2 model, run the way it was tested.
    if args.weights is None and args.model_type is None and args.preset is None:
        args.preset = "synthetic"
    if args.threshold is None:
        args.threshold = "auto" if args.preset else "0.5"
    if args.model_type is None:
        args.model_type = "dinov2_lc" if args.mode == "semantic" else "dinov2"
    if args.weights is None:
        args.weights = DEFAULT_WEIGHTS

    preset = PRESETS.get(args.preset, {}) if args.preset else {}
    for key, fallback in (("scale", 1.0), ("tta", False), ("logit_adjust", False),
                          ("target_gsd", 0.0), ("tile_size", 256), ("overlap", 0),
                          ("core_threshold", 0.0), ("min_blob_m2", 0.0), ("min_width_m", 0.0)):
        if getattr(args, key) is None:
            setattr(args, key, preset.get(key, fallback))
    if args.fast:
        args.tta = False

    return preset


def main():
    global _json_progress

    args = build_parser().parse_args()
    preset = resolve_settings(args)

    _json_progress = args.json_progress

    # Labelled change runs the binary pipeline, then names each change blob.
    labelled = args.model_type == "dinov2_lc"
    if labelled:
        args.mode = "binary"

    if args.no_cleanup:
        args.core_threshold = args.min_blob_m2 = args.min_width_m = 0.0

    if args.threshold == "auto" and "threshold" in preset:
        args.auto_threshold = False
        args.threshold = preset_threshold(preset, args.tta)
        fast = ""
        if preset.get("tta") and not args.tta:
            fast = (" (fast mode: no flip averaging)" if "fast_threshold" in preset else
                    " (fast mode: no flip averaging; threshold tested with flip averaging)")
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

    from uchange_qgis_plugin.model_bridge import build_model

    project_root = os.path.dirname(os.path.abspath(__file__))
    weights_path = args.weights
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(project_root, weights_path)
    if not os.path.isfile(weights_path):
        _emit({"type": "error", "message": f"Model weights not found: {weights_path}"})
        sys.exit(1)
    model_type = args.model_type

    if model_type in ("dinov2", "ensemble", "dinov2_lc"):
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
        model_type="dinov2" if model_type in ("ensemble", "dinov2_lc") else model_type)
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
        except ImportError:
            _log("  AROSICS not available. Skipping co-registration.")
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
    thresh_str = "auto" if args.auto_threshold else f"{args.threshold}"
    _log(f"Tile size: {args.tile_size}, overlap: {args.overlap}, threshold: {thresh_str}")

    # Resolution: the model's own scale, times an upsampling for coarse imagery.
    from uchange_qgis_plugin.resolution import working_factor, resample_rgb, resample_prob
    factor, gsd, gsd_note = (1.0, None, "")
    if args.mode != "semantic":
        factor, gsd, gsd_note = working_factor(before_geo or after_geo, args.target_gsd,
                                               args.scale, before_img.shape)
        if gsd:
            _log(f"Pixel size: {gsd:.2f} m")
        if gsd_note:
            _log(f"  {gsd_note}")
        if abs(factor - 1.0) > 1e-6:
            _log(f"Inference at x{factor:.2f} ({int(w * factor)}x{int(h * factor)} px)")
        if args.tta or args.logit_adjust:
            _log(f"Flip averaging: {'on' if args.tta else 'off'}, "
                 f"logit adjustment: {'on' if args.logit_adjust else 'off'}")

    os.makedirs(args.output, exist_ok=True)

    from uchange_qgis_plugin.tiling import run_tiled_inference

    t0 = time.time()
    _log(f"Running {'SCD' if args.mode == 'semantic' else 'binary'} tiled inference...")

    def progress_fn(current, total):
        pct = 20 + int(60 * current / total)
        _progress(pct)

    if args.histogram_match:
        _log("Histogram matching: per-tile (after → before)")

    try:
        result = run_tiled_inference(
            model, resample_rgb(before_img, factor), resample_rgb(after_img, factor),
            tile_size=args.tile_size,
            overlap=args.overlap,
            device=device,
            grayscale=args.grayscale,
            progress_fn=progress_fn,
            hist_match=args.histogram_match,
            tta=args.tta,
            logit_adjust=args.logit_adjust,
        )
        if args.mode != "semantic":
            result = resample_prob(result, (h, w))
        if cs_model is not None:
            # ChangeStar was evaluated at native scale: only the GSD correction applies.
            from uchange_qgis_plugin.changestar_bridge import changestar_probs
            cs_factor = factor / args.scale
            _log("Running Changen2 ChangeStar ViT-L...")
            p_cs = changestar_probs(cs_model, resample_rgb(before_img, cs_factor),
                                    resample_rgb(after_img, cs_factor), device)
            p_cs = resample_prob(p_cs, (h, w))
            # Geometric mean: a pixel is change only if both models lean that way.
            result = np.sqrt(np.clip(result, 0, 1) * np.clip(p_cs, 0, 1)).astype(np.float32)
            del cs_model
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            _emit({"type": "error", "message": f"GPU out of memory. Try --tile-size 128 or --device cpu."})
            sys.exit(1)
        raise
    elapsed = time.time() - t0
    _log(f"Inference: {elapsed:.1f}s")

    if not labelled:
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

    if args.mode == "semantic":
        prob_map = result['prob_map']
        semantic_to = result['semantic_to']

        binary_mask = (prob_map > 0.5).astype(np.uint8)
        n_change = int(binary_mask.sum())
        total = binary_mask.size
        _log(f"Change pixels: {n_change}/{total} ({n_change/total:.2%})")

        from uchange_qgis_plugin.model_bridge import SECOND_SEMANTIC_CLASSES, SECOND_SEMANTIC_PALETTE

        num_classes = len(SECOND_SEMANTIC_CLASSES)
        sem_to_masked = np.clip((semantic_to.astype(np.int16) + 1) * binary_mask, 0, num_classes - 1).astype(np.uint8)

        result_info = {
            "type": "result",
            "mode": "semantic",
            "stats": {"change_pixels": n_change, "total_pixels": total},
        }

        if geo_info:
            from uchange_qgis_plugin.raster_io import save_semantic_geotiff, save_binary_geotiff
            binary_path = os.path.join(args.output, "binary_change.tif")
            semantic_path = os.path.join(args.output, "semantic_change.tif")
            save_binary_geotiff(binary_mask, geo_info["geotransform"], geo_info["projection"], binary_path)
            save_semantic_geotiff(sem_to_masked, geo_info["geotransform"], geo_info["projection"],
                                  semantic_path,
                                  SECOND_SEMANTIC_CLASSES, SECOND_SEMANTIC_PALETTE)
            _log(f"Saved: {binary_path} (georeferenced)")
            _log(f"Saved: {semantic_path} (georeferenced)")
            result_info["binary_path"] = binary_path
            result_info["semantic_path"] = semantic_path

        palette = np.array(SECOND_SEMANTIC_PALETTE, dtype=np.uint8)
        colored = palette[sem_to_masked]
        png_path = os.path.join(args.output, "semantic_change.png")
        Image.fromarray(colored).save(png_path)
        _log(f"Saved: {png_path}")

        vis = np.concatenate([before_img, after_img, colored], axis=1)
        comparison_path = os.path.join(args.output, "comparison.png")
        Image.fromarray(vis).save(comparison_path)
        _log(f"Saved: {comparison_path}")

        _progress(100)
        _emit(result_info)

    else:
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
        binary_mask = clean_mask(prob_map, args.threshold, args.core_threshold or None,
                                 area_px, width_px)
        if args.core_threshold or args.min_blob_m2 or args.min_width_m:
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
                geo_info, factor / args.scale))
        elif args.output_gpkg and geo_info:
            _log(f"Polygonizing to {args.output_gpkg}...")
            from uchange_qgis_plugin.raster_io import polygonize_mask

            pixel_w = abs(geo_info["geotransform"][1])
            pixel_h = abs(geo_info["geotransform"][5])
            min_area_map = args.min_area * pixel_w * pixel_h

            style = "convex hull" if args.style == "convex" else args.style
            total_polys, final_polys = polygonize_mask(
                binary_mask, geo_info["geotransform"], geo_info["projection"],
                args.output_gpkg, min_area=min_area_map, style=style,
            )
            _log(f"Polygons: {total_polys} created, {final_polys} after min-area filter")
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

        mask_vis = np.stack([binary_mask * 255] * 3, axis=-1).astype(np.uint8)
        vis = np.concatenate([before_img, after_img, mask_vis], axis=1)
        comparison_path = os.path.join(args.output, "comparison.png")
        Image.fromarray(vis).save(comparison_path)
        _log(f"Saved: {comparison_path}")

        prob_vis = (prob_map * 255).clip(0, 255).astype(np.uint8)
        prob_path = os.path.join(args.output, "probability.png")
        Image.fromarray(prob_vis).save(prob_path)
        _log(f"Saved: {prob_path}")

        _progress(100)
        _emit(result_info)

    if coreg_cleanup:
        coreg_cleanup()

    if not _json_progress:
        print(f"\nDone! Results in {args.output}/")


if __name__ == "__main__":
    main()
