"""
Benchmark CD model speed and VRAM at increasing image sizes.

Stitches LEVIR-CD test pairs into mosaics (1024, 2048, 4096, 8192)
and measures inference time + peak VRAM at each scale. No accuracy
metrics — use benchmark_cd.py for those.

Usage:
    python benchmark_scale.py \
        --data-root /path/to/LEVIR-CD/test \
        --weights-dir /path/to/weights
    python benchmark_scale.py --models ChangerEx_r18 CGNet
"""

import os
import gc
import time
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Model registry (same as benchmark_cd.py)
# ---------------------------------------------------------------------------

OPENCD_MODELS = [
    {"name": "ChangerEx_r18",    "config": "configs/changer/changer_ex_r18_512x512_40k_levircd.py",              "ckpt": "ChangerEx_r18-512x512_40k_levircd.pth"},
    {"name": "BAN_vit-l14",      "config": "configs/ban/ban_vit-l14-clip_mit-b0_512x512_40k_levircd.py",         "ckpt": "ban_vit-l14-clip_mit-b0_512x512_40k_levircd.pth"},
    {"name": "BiT_r18",          "config": "configs/bit/bit_r18_256x256_40k_levircd.py",                         "ckpt": "bit_r18_256x256_40k_levircd.pth"},
    {"name": "CGNet",            "config": "configs/cgnet/cgnet_256x256_40k_levircd.py",                         "ckpt": "cgnet_256x256_40k_levircd.pth"},
    {"name": "ChangeFormer_b0",  "config": "configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py",    "ckpt": "changeformer_mit-b0_256x256_40k_levircd.pth"},
    {"name": "ChangeFormer_b1",  "config": "configs/changeformer/changeformer_mit-b1_256x256_40k_levircd.py",    "ckpt": "changeformer_mit-b1_256x256_40k_levircd.pth"},
    {"name": "ChangeStar",       "config": "configs/changestar/changestar_farseg_1x96_512x512_40k_levircd.py",   "ckpt": "changestar_farseg_1x96_512x512_40k_levircd.pth"},
    {"name": "FC-EF",            "config": "configs/fcsn/fc_ef_256x256_40k_levircd.py",                          "ckpt": "fc_ef_256x256_40k_levircd.pth"},
    {"name": "FC-Siam-Conc",     "config": "configs/fcsn/fc_siam_conc_256x256_40k_levircd.py",                   "ckpt": "fc_siam_conc_256x256_40k_levircd.pth"},
    {"name": "FC-Siam-Diff",     "config": "configs/fcsn/fc_siam_diff_256x256_40k_levircd.py",                   "ckpt": "fc_siam_diff_256x256_40k_levircd.pth"},
    {"name": "HANet",            "config": "configs/hanet/hanet_256x256_40k_levircd.py",                         "ckpt": "hanet_256x256_40k_levircd.pth"},
    {"name": "IFN",              "config": "configs/ifn/ifn_256x256_40k_levircd.py",                             "ckpt": "ifn_256x256_40k_levircd.pth"},
    {"name": "LightCDNet_b",     "config": "configs/lightcdnet/lightcdnet_b_256x256_40k_levircd.py",             "ckpt": "lightcdnet_b_256x256_40k_levircd.pth"},
    {"name": "LightCDNet_l",     "config": "configs/lightcdnet/lightcdnet_l_256x256_40k_levircd.py",             "ckpt": "lightcdnet_l_256x256_40k_levircd.pth"},
    {"name": "SNUNet",           "config": "configs/snunet/snunet_c16_256x256_40k_levircd.py",                   "ckpt": "snunet_c16_256x256_40k_levircd.pth"},
    {"name": "STANet_PAM",       "config": "configs/stanet/stanet_pam_256x256_40k_levircd.py",                   "ckpt": "stanet_pam_256x256_40k_levircd.pth"},
]

SCALES = [1024, 2048, 4096, 8192]

# ---------------------------------------------------------------------------
# Mosaic creation
# ---------------------------------------------------------------------------

def load_source_images(data_root, max_images=16):
    a_dir = os.path.join(data_root, "A")
    b_dir = os.path.join(data_root, "B")
    fnames = sorted(
        f for f in os.listdir(a_dir)
        if f.lower().endswith((".png", ".jpg", ".tif"))
    )[:max_images]

    pairs = []
    for fname in fnames:
        img_a = cv2.imread(os.path.join(a_dir, fname))
        img_b = cv2.imread(os.path.join(b_dir, fname))
        if img_a is not None and img_b is not None:
            pairs.append((img_a, img_b))
    return pairs


def create_mosaic(source_pairs, target_size):
    src_h, src_w = source_pairs[0][0].shape[:2]
    grid = target_size // src_h

    mosaic_a = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    mosaic_b = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    idx = 0
    for row in range(grid):
        for col in range(grid):
            img_a, img_b = source_pairs[idx % len(source_pairs)]
            r0, r1 = row * src_h, (row + 1) * src_h
            c0, c1 = col * src_w, (col + 1) * src_w
            mosaic_a[r0:r1, c0:c1] = img_a
            mosaic_b[r0:r1, c0:c1] = img_b
            idx += 1

    return mosaic_a, mosaic_b

# ---------------------------------------------------------------------------
# Model loading (reuses benchmark_cd.py patterns)
# ---------------------------------------------------------------------------

def _neutralize_pretrained(cfg):
    if hasattr(cfg.model, "pretrained"):
        cfg.model.pretrained = None
    for attr in ("backbone", "image_encoder", "decode_head", "auxiliary_head"):
        sub = getattr(cfg.model, attr, None)
        if sub is not None and hasattr(sub, "init_cfg"):
            sub.init_cfg = None


def build_opencd_model(cfg_path, ckpt_path, device):
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import DefaultScope
    import opencd.models  # noqa: F401

    cfg = Config.fromfile(os.path.join(PROJECT_ROOT, cfg_path))
    _neutralize_pretrained(cfg)

    from opencd.registry import MODELS
    DefaultScope.get_instance('opencd', scope_name='opencd')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location="cpu", logger=None)
    model = revert_sync_batchnorm(model)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Inference at a given scale
# ---------------------------------------------------------------------------

def run_opencd_at_scale(model, mosaic_a_bgr, mosaic_b_bgr, device):
    from mmseg.structures import SegDataSample

    a_chw = torch.from_numpy(mosaic_a_bgr.transpose(2, 0, 1).copy())
    b_chw = torch.from_numpy(mosaic_b_bgr.transpose(2, 0, 1).copy())
    tensor_6ch = torch.cat([a_chw, b_chw], dim=0)

    h, w = mosaic_a_bgr.shape[:2]
    data_sample = SegDataSample()
    data_sample.set_metainfo({
        "ori_shape": (h, w),
        "img_shape": (h, w),
        "pad_shape": (h, w),
        "padding_size": [0, 0, 0, 0],
    })

    data = {"inputs": [tensor_6ch], "data_samples": [data_sample]}

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        model.test_step(data)

    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAX_MS = 60000


def benchmark_model_at_scale(model, mosaic_a, mosaic_b, device):
    probe = run_opencd_at_scale(model, mosaic_a, mosaic_b, device)
    if probe > MAX_MS:
        return None  # too slow, discard
    if probe > 10000:
        return probe
    times = [probe]
    for _ in range(2):
        times.append(run_fn(model, mosaic_a, mosaic_b, device))
    return np.mean(times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark CD model scaling with image size")
    parser.add_argument("--data-root", required=True, help="LEVIR-CD test dir with A/, B/")
    parser.add_argument("--weights-dir", required=True, help="Directory with .pth/.ckpt files")
    parser.add_argument("--models", nargs="*", default=None, help="Run only these models")
    parser.add_argument("--scales", nargs="*", type=int, default=SCALES, help="Image sizes to test (default: 1024 2048 4096 8192)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"Loading source images from {args.data_root} ...")
    source_pairs = load_source_images(args.data_root, max_images=64)
    print(f"Loaded {len(source_pairs)} source pairs")

    scales = sorted(args.scales)
    print(f"Scales to test: {scales}")

    print("Creating mosaics ...")
    mosaics = {}
    for s in scales:
        mosaics[s] = create_mosaic(source_pairs, s)
        print(f"  {s}x{s} mosaic ready")

    all_models = OPENCD_MODELS
    if args.models:
        selected = set(args.models)
        all_models = [m for m in all_models if m["name"] in selected]

    # results[model_name][scale] = {"time_ms": ..., "vram_mb": ...} or "OOM"
    results = {}

    for mi, model_info in enumerate(all_models, 1):
        name = model_info["name"]
        ckpt_path = os.path.join(args.weights_dir, model_info["ckpt"])

        print(f"\n{'=' * 60}")
        print(f"[{mi}/{len(all_models)}] {name}")
        print(f"{'=' * 60}")

        if not os.path.isfile(ckpt_path):
            print(f"  SKIP: checkpoint not found")
            results[name] = {s: "SKIP" for s in scales}
            continue

        try:
            model = build_opencd_model(model_info["config"], ckpt_path, device)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            results[name] = {s: "FAIL" for s in scales}
            continue

        results[name] = {}

        for scale in scales:
            mosaic_a, mosaic_b = mosaics[scale]

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            print(f"  {scale}x{scale}: ", end="", flush=True)

            try:
                avg_time = benchmark_model_at_scale(
                    model, mosaic_a, mosaic_b, device
                )
                if avg_time is None:
                    print(f">60s — skipping larger scales")
                    results[name][scale] = "SLOW"
                    for remaining_scale in scales[scales.index(scale) + 1:]:
                        results[name][remaining_scale] = "SLOW"
                    break

                vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0
                results[name][scale] = {"time_ms": avg_time, "vram_mb": vram}
                print(f"{avg_time:.0f}ms  {vram:.0f}MB")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM")
                    results[name][scale] = "OOM"
                    torch.cuda.empty_cache()
                    for remaining_scale in scales[scales.index(scale) + 1:]:
                        results[name][remaining_scale] = "OOM"
                        print(f"  {remaining_scale}x{remaining_scale}: OOM (skipped)")
                    break
                else:
                    print(f"ERROR: {e}")
                    results[name][scale] = "FAIL"

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Print results table
    print(f"\n{'=' * 100}")
    print("SCALING BENCHMARK RESULTS")
    print(f"{'=' * 100}")

    name_w = max(len(n) for n in results) + 2
    header = f"{'Model':<{name_w}}"
    for s in scales:
        header += f"  {'Time ' + str(s):>12}  {'VRAM ' + str(s):>12}"
    print(header)
    print("-" * len(header))

    for name in results:
        row = f"{name:<{name_w}}"
        for s in scales:
            entry = results[name].get(s, "N/A")
            if isinstance(entry, dict):
                row += f"  {entry['time_ms']:>10.0f}ms  {entry['vram_mb']:>10.0f}MB"
            else:
                row += f"  {entry:>12}  {entry:>12}"
        print(row)

    print(f"{'=' * 100}")

    # Show which models survive at each scale
    print(f"\nModels that fit in VRAM at each scale:")
    for s in scales:
        survivors = []
        for name in results:
            entry = results[name].get(s)
            if isinstance(entry, dict):
                survivors.append((name, entry["time_ms"], entry["vram_mb"]))
        survivors.sort(key=lambda x: x[1])
        if survivors:
            print(f"\n  {s}x{s}: {len(survivors)} models fit")
            for rank, (n, t, v) in enumerate(survivors, 1):
                print(f"    {rank:>2}. {n:<22} {t:>8.0f}ms  {v:>8.0f}MB")
        else:
            print(f"\n  {s}x{s}: all models OOM")


if __name__ == "__main__":
    main()
