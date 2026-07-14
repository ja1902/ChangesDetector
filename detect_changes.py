"""
Run ChangerEx change detection on your own image pair.

Usage:
    python detect_changes.py --before path/to/before.tif --after path/to/after.tif
    python detect_changes.py --before before.png --after after.png --output my_result
    python detect_changes.py --before before.tif --after after.tif --threshold 0.3 --tile-size 256
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
                img = np.stack(bands, axis=-1).astype(np.uint8)
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


def run_inference(model, before_img, after_img, tile_size, device, overlap=0):
    from uchange_qgis_plugin.model_bridge import normalize_tile

    h, w = before_img.shape[:2]
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    step = tile_size - overlap
    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = min(y + tile_size, h)
            x1 = min(x + tile_size, w)
            y0 = max(0, y1 - tile_size)
            x0 = max(0, x1 - tile_size)
            tiles.append((y0, x0, y1, x1))

    batch_size = 16
    use_amp = device.type == "cuda"

    for start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[start:start + batch_size]
        pre_list, post_list, meta = [], [], []

        for y0, x0, y1, x1 in batch_tiles:
            pre_tile = before_img[y0:y1, x0:x1]
            post_tile = after_img[y0:y1, x0:x1]
            th, tw = pre_tile.shape[:2]

            if th < tile_size or tw < tile_size:
                padded_pre = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded_post = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded_pre[:th, :tw] = pre_tile
                padded_post[:th, :tw] = post_tile
                pre_tile, post_tile = padded_pre, padded_post

            pre_list.append(normalize_tile(pre_tile))
            post_list.append(normalize_tile(post_tile))
            meta.append((y0, x0, y1, x1, th, tw))

        pre_batch = torch.from_numpy(np.stack(pre_list)).float().to(device)
        post_batch = torch.from_numpy(np.stack(post_list)).float().to(device)

        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            logits = model(pre_batch, post_batch)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()

        for j, (y0, x0, y1, x1, th, tw) in enumerate(meta):
            prob_map[y0:y1, x0:x1] += probs[j, :th, :tw]
            count_map[y0:y1, x0:x1] += 1.0

        done = min(start + batch_size, len(tiles))
        print(f"  Tiles: {done}/{len(tiles)}", end="\r")

    print()
    count_map = np.maximum(count_map, 1.0)
    return (prob_map / count_map).astype(np.float32)


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
    parser = argparse.ArgumentParser(description="ChangerEx change detection on custom images")
    parser.add_argument("--before", required=True, help="Path to before image")
    parser.add_argument("--after", required=True, help="Path to after image")
    parser.add_argument("--weights", default="ChangerEx_r18-512x512_40k_levircd.pth")
    parser.add_argument("--output", default="change_result", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5)
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
    model, load_summary = build_model(weights_path, device)
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

    t0 = time.time()
    prob_map = run_inference(model, before_img, after_img, args.tile_size, device, args.overlap)
    elapsed = time.time() - t0
    print(f"Inference: {elapsed:.1f}s")

    print(f"Prob map: min={prob_map.min():.4f}, max={prob_map.max():.4f}, mean={prob_map.mean():.4f}")

    binary_mask = (prob_map > args.threshold).astype(np.uint8)
    n_change = int(binary_mask.sum())
    total = binary_mask.size
    print(f"Change pixels: {n_change}/{total} ({n_change/total:.2%})")

    geo_info = before_geo or after_geo

    if geo_info:
        save_geotiff(os.path.join(args.output, "mask.tif"), binary_mask * 255, geo_info)
        print(f"Saved: {args.output}/mask.tif (georeferenced)")
    else:
        Image.fromarray(binary_mask * 255).save(os.path.join(args.output, "mask.png"))
        print(f"Saved: {args.output}/mask.png")

    # Visualization: before | after | mask side by side
    mask_vis = np.stack([binary_mask * 255] * 3, axis=-1).astype(np.uint8)
    vis = np.concatenate([before_img, after_img, mask_vis], axis=1)
    Image.fromarray(vis).save(os.path.join(args.output, "comparison.png"))
    print(f"Saved: {args.output}/comparison.png")

    # Probability heatmap (grayscale)
    prob_vis = (prob_map * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(prob_vis).save(os.path.join(args.output, "probability.png"))
    print(f"Saved: {args.output}/probability.png")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if coreg_cleanup:
        coreg_cleanup()

    print(f"\nDone! Results in {args.output}/")


if __name__ == "__main__":
    main()
