"""
Run change detection on your own image pair.

Usage:
    python detect_changes.py --before path/to/before.tif --after path/to/after.tif
    python detect_changes.py --before before.png --after after.png --output my_result
    python detect_changes.py --before before.tif --after after.tif --threshold 0.3 --tile-size 256
    python detect_changes.py --before before.tif --after after.tif --mode semantic --weights scd_upernet_r18_10k_second.pth
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

import torch


def read_image(path):
    """Read image as RGB numpy array. Supports PNG/JPG/TIF via PIL, or georeferenced via GDAL."""
    ext = os.path.splitext(path)[1].lower()

    geo_info = None

    if ext in (".tif", ".tiff"):
        try:
            from osgeo import gdal
            gdal.UseExceptions()
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds is not None:
                bands = []
                for i in range(1, min(ds.RasterCount, 3) + 1):
                    bands.append(ds.GetRasterBand(i).ReadAsArray())
                img = np.stack(bands, axis=-1)
                if img.dtype != np.uint8:
                    max_val = img.max()
                    if max_val > 255:
                        print(f"  Warning: {img.dtype} image detected, rescaling to 8-bit")
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
        except Exception:
            pass

    img = np.array(Image.open(path).convert("RGB"))
    return img, geo_info


def save_geotiff(path, array, geo_info):
    """Save a single-band array as GeoTIFF with georeferencing."""
    from osgeo import gdal
    gdal.UseExceptions()
    h, w = array.shape
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, w, h, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(geo_info["geotransform"])
    ds.SetProjection(geo_info["projection"])
    ds.GetRasterBand(1).WriteArray(array)
    ds.FlushCache()
    ds = None


def main():
    parser = argparse.ArgumentParser(description="Change detection on custom images")
    parser.add_argument("--before", required=True, help="Path to before image")
    parser.add_argument("--after", required=True, help="Path to after image")
    parser.add_argument("--weights", default=None,
                        help="Path to model weights (default depends on --mode)")
    parser.add_argument("--mode", choices=["binary", "semantic"], default="binary",
                        help="Detection mode: binary (default) or semantic")
    parser.add_argument("--output", default="change_result", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Change threshold 0.0-1.0 (default: 0.5)")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-coreg", action="store_true",
                        help="Skip co-registration of input images")
    parser.add_argument("--max-shift", type=int, default=50,
                        help="Max co-registration shift in pixels (default: 50)")
    parser.add_argument("--coreg-window", type=int, default=1024,
                        help="Co-registration matching window size (default: 1024)")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        print(f"WARNING: threshold {args.threshold} outside [0, 1], clamping")
        args.threshold = max(0.0, min(1.0, args.threshold))

    for path, label in [(args.before, "Before image"), (args.after, "After image")]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    if args.weights is None:
        if args.mode == "semantic":
            args.weights = "scd_upernet_r18_10k_second.pth"
        else:
            args.weights = "ChangerEx_r18-512x512_40k_levircd.pth"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    from uchange_qgis_plugin.model_bridge import build_model

    project_root = os.path.dirname(os.path.abspath(__file__))
    weights_path = args.weights
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(project_root, weights_path)
    if not os.path.isfile(weights_path):
        print(f"ERROR: Model weights not found: {weights_path}")
        print("Run the installer to download weights, or use --weights to specify a path.")
        sys.exit(1)
    model_type = "opencd_scd" if args.mode == "semantic" else "opencd"
    model, load_summary = build_model(weights_path, device, model_type=model_type)
    print(load_summary)

    print(f"\nReading before: {args.before}")
    before_img, before_geo = read_image(args.before)
    print(f"Reading after:  {args.after}")
    after_img, after_geo = read_image(args.after)

    coreg_cleanup = None
    if not args.no_coreg and before_geo and after_geo:
        print("Co-registering images...")
        try:
            from uchange_qgis_plugin.coregistration import coregister_images
            coreg_result = coregister_images(
                args.before, args.after,
                max_shift=args.max_shift,
                window_size=(args.coreg_window, args.coreg_window),
            )
            if coreg_result.success and coreg_result.corrected_path:
                print(f"  Shift: X={coreg_result.shift_x_px:.2f}px, Y={coreg_result.shift_y_px:.2f}px")
                after_img, after_geo = read_image(coreg_result.corrected_path)
                coreg_cleanup = coreg_result.cleanup
            elif coreg_result.success:
                print(f"  {coreg_result.message}")
            else:
                msg = coreg_result.message.rstrip('.')
                print(f"  {msg}. Using original images.")
        except ImportError:
            print("  AROSICS not available. Skipping co-registration.")
        except Exception as e:
            print(f"  Co-registration failed: {e}. Using original images.")

    if before_img.shape[:2] != after_img.shape[:2]:
        bh, bw = before_img.shape[:2]
        ah, aw = after_img.shape[:2]
        if abs(bh - ah) <= 2 and abs(bw - aw) <= 2:
            h_min, w_min = min(bh, ah), min(bw, aw)
            print(f"  Trimming to common size: {w_min}x{h_min} (was {bw}x{bh} / {aw}x{ah})")
            before_img = before_img[:h_min, :w_min]
            after_img = after_img[:h_min, :w_min]
        else:
            print(f"ERROR: Image dimensions don't match: before={before_img.shape[:2]}, after={after_img.shape[:2]}")
            sys.exit(1)

    h, w = before_img.shape[:2]
    print(f"Image size: {w}x{h}")
    print(f"Tile size: {args.tile_size}, overlap: {args.overlap}, threshold: {args.threshold}")

    os.makedirs(args.output, exist_ok=True)

    from uchange_qgis_plugin.tiling import run_tiled_inference

    t0 = time.time()
    print(f"Running {'SCD' if args.mode == 'semantic' else 'binary'} tiled inference...")

    try:
        result = run_tiled_inference(
            model, before_img, after_img,
            tile_size=args.tile_size,
            overlap=args.overlap,
            device=device,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nERROR: GPU ran out of memory.")
            print(f"Try reducing tile size (current: {args.tile_size}):")
            print(f"  --tile-size 128")
            print(f"Or switch to CPU (slower but no VRAM limit):")
            print(f"  --device cpu")
            sys.exit(1)
        raise
    elapsed = time.time() - t0
    print(f"Inference: {elapsed:.1f}s")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    geo_info = before_geo or after_geo

    if args.mode == "semantic":
        prob_map = result['prob_map']
        semantic_to = result['semantic_to']

        binary_mask = (prob_map > 0.5).astype(np.uint8)
        n_change = int(binary_mask.sum())
        total = binary_mask.size
        print(f"Change pixels: {n_change}/{total} ({n_change/total:.2%})")

        from uchange_qgis_plugin.model_bridge import SECOND_SEMANTIC_CLASSES, SECOND_SEMANTIC_PALETTE

        # cover_semantic: shift class indices +1, mask by binary change, clamp to valid range
        num_classes = len(SECOND_SEMANTIC_CLASSES)
        sem_to_masked = np.clip((semantic_to.astype(np.int16) + 1) * binary_mask, 0, num_classes - 1).astype(np.uint8)

        if geo_info:
            from uchange_qgis_plugin.raster_io import save_semantic_geotiff, save_binary_geotiff
            save_binary_geotiff(binary_mask, geo_info["geotransform"], geo_info["projection"], os.path.join(args.output, "binary_change.tif"))
            save_semantic_geotiff(sem_to_masked, geo_info["geotransform"], geo_info["projection"],
                                  os.path.join(args.output, "semantic_change.tif"),
                                  SECOND_SEMANTIC_CLASSES, SECOND_SEMANTIC_PALETTE)
            print(f"Saved: {args.output}/binary_change.tif (georeferenced)")
            print(f"Saved: {args.output}/semantic_change.tif (georeferenced)")

        palette = np.array(SECOND_SEMANTIC_PALETTE, dtype=np.uint8)
        colored = palette[sem_to_masked]
        Image.fromarray(colored).save(os.path.join(args.output, "semantic_change.png"))
        print(f"Saved: {args.output}/semantic_change.png")

        # Comparison: before | after | semantic change
        vis = np.concatenate([before_img, after_img, colored], axis=1)
        Image.fromarray(vis).save(os.path.join(args.output, "comparison.png"))
        print(f"Saved: {args.output}/comparison.png")

    else:
        prob_map = result
        print(f"Prob map: min={prob_map.min():.4f}, max={prob_map.max():.4f}, mean={prob_map.mean():.4f}")

        binary_mask = (prob_map > args.threshold).astype(np.uint8)
        n_change = int(binary_mask.sum())
        total = binary_mask.size
        print(f"Change pixels: {n_change}/{total} ({n_change/total:.2%})")

        if geo_info:
            save_geotiff(os.path.join(args.output, "mask.tif"), binary_mask * 255, geo_info)
            print(f"Saved: {args.output}/mask.tif (georeferenced)")
        else:
            Image.fromarray(binary_mask * 255).save(os.path.join(args.output, "mask.png"))
            print(f"Saved: {args.output}/mask.png")

        mask_vis = np.stack([binary_mask * 255] * 3, axis=-1).astype(np.uint8)
        vis = np.concatenate([before_img, after_img, mask_vis], axis=1)
        Image.fromarray(vis).save(os.path.join(args.output, "comparison.png"))
        print(f"Saved: {args.output}/comparison.png")

        prob_vis = (prob_map * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(prob_vis).save(os.path.join(args.output, "probability.png"))
        print(f"Saved: {args.output}/probability.png")

    if coreg_cleanup:
        coreg_cleanup()

    print(f"\nDone! Results in {args.output}/")


if __name__ == "__main__":
    main()
