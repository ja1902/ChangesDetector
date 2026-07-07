"""
Benchmark all change detection models on LEVIR-CD test set.

Usage:
    export DATA_ROOT=/path/to/LEVIR-CD/test   # must contain A/, B/, label/
    export WEIGHTS_DIR=/path/to/weights        # all .pth files
    python benchmark_cd.py
    python benchmark_cd.py --models ChangerEx_r18   # run specific models only

Outputs:
    - Printed comparison table
    - benchmark_results.csv
"""

import os
import sys
import gc
import csv
import time
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Model registry
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(data_root):
    a_dir = os.path.join(data_root, "A")
    b_dir = os.path.join(data_root, "B")
    label_dir = os.path.join(data_root, "label")

    for d in (a_dir, b_dir, label_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing directory: {d}")

    fnames = sorted(
        f for f in os.listdir(a_dir)
        if f.lower().endswith((".png", ".jpg", ".tif"))
    )
    if not fnames:
        raise FileNotFoundError(f"No images found in {a_dir}")

    data = []
    for fname in fnames:
        img_a = cv2.imread(os.path.join(a_dir, fname))
        img_b = cv2.imread(os.path.join(b_dir, fname))
        label = cv2.imread(os.path.join(label_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img_a is None or img_b is None or label is None:
            print(f"  Warning: could not read {fname}, skipping")
            continue
        label_bin = (label >= 128).astype(np.uint8)
        data.append((img_a, img_b, label_bin, fname))

    return data

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def boundary_mask(binary_mask, d=3):
    """Pixels within d pixels of a label boundary."""
    if binary_mask.sum() == 0 or binary_mask.all():
        return np.zeros_like(binary_mask, dtype=bool)
    eroded = ndimage.binary_erosion(binary_mask, iterations=d)
    dilated = ndimage.binary_dilation(binary_mask, iterations=d)
    return dilated & ~eroded


def compute_metrics(all_preds, all_gts, boundary_d=3, small_thresh=100):
    """Compute all 6 accuracy metrics from lists of prediction and GT masks."""
    total_tp = total_fp = total_fn = 0
    bnd_tp = bnd_fp = bnd_fn = 0
    small_tp = small_fn = 0

    for pred, gt in zip(all_preds, all_gts):
        pred_bool = pred.astype(bool)
        gt_bool = gt.astype(bool)

        tp = (pred_bool & gt_bool).sum()
        fp = (pred_bool & ~gt_bool).sum()
        fn = (~pred_bool & gt_bool).sum()
        total_tp += tp
        total_fp += fp
        total_fn += fn

        gt_bnd = boundary_mask(gt_bool, boundary_d)
        if gt_bnd.any():
            bnd_tp += (pred_bool & gt_bool & gt_bnd).sum()
            bnd_fp += (pred_bool & ~gt_bool & gt_bnd).sum()
            bnd_fn += (~pred_bool & gt_bool & gt_bnd).sum()

        labeled, num_features = ndimage.label(gt_bool)
        for comp_id in range(1, num_features + 1):
            comp_mask = labeled == comp_id
            area = comp_mask.sum()
            if area < small_thresh:
                comp_tp = (pred_bool & comp_mask).sum()
                comp_fn = (~pred_bool & comp_mask).sum()
                small_tp += comp_tp
                small_fn += comp_fn

    eps = 1e-8
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)

    bnd_prec = bnd_tp / (bnd_tp + bnd_fp + eps)
    bnd_rec = bnd_tp / (bnd_tp + bnd_fn + eps)
    bnd_f1 = 2 * bnd_prec * bnd_rec / (bnd_prec + bnd_rec + eps)

    small_rec = small_tp / (small_tp + small_fn + eps) if (small_tp + small_fn) > 0 else float("nan")

    return {
        "F1": float(f1),
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "BF1": float(bnd_f1),
        "SmallRec": float(small_rec),
    }

# ---------------------------------------------------------------------------
# Open-CD model inference
# ---------------------------------------------------------------------------

def _neutralize_pretrained(cfg):
    """Remove pretrained/init_cfg paths that would trigger downloads."""
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
    import opencd.models  # noqa: F401 — triggers registry
    from mmengine.registry import DefaultScope

    cfg = Config.fromfile(os.path.join(PROJECT_ROOT, cfg_path))
    _neutralize_pretrained(cfg)

    from opencd.registry import MODELS
    DefaultScope.get_instance('opencd', scope_name='opencd')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location="cpu", logger=None)
    model = revert_sync_batchnorm(model)
    model.to(device).eval()
    return model


def run_opencd_inference(model, test_data, device, warmup=3):
    """Run Open-CD model on all test images. Returns (preds, times_ms)."""
    from mmseg.structures import SegDataSample

    all_preds = []
    times_ms = []

    for idx, (img_a_bgr, img_b_bgr, _, _) in enumerate(test_data):
        a_chw = torch.from_numpy(img_a_bgr.transpose(2, 0, 1).copy())
        b_chw = torch.from_numpy(img_b_bgr.transpose(2, 0, 1).copy())
        tensor_6ch = torch.cat([a_chw, b_chw], dim=0)  # (6, H, W) uint8 BGR

        h, w = img_a_bgr.shape[:2]
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
            results = model.test_step(data)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0

        pred = results[0].pred_sem_seg.data.cpu().numpy().squeeze().astype(np.uint8)
        all_preds.append(pred)

        if idx >= warmup:
            times_ms.append(elapsed)

        print(f"\r  Inference: {idx + 1}/{len(test_data)}", end="", flush=True)

    print()
    return all_preds, times_ms


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

METRIC_KEYS = ["F1", "IoU", "Precision", "Recall", "BF1", "SmallRec", "Time_ms", "VRAM_MB"]

def print_table(results):
    name_w = max(len(r["name"]) for r in results) + 2
    header = f"{'Model':<{name_w}}"
    for k in METRIC_KEYS:
        header += f"  {k:>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        row = f"{r['name']:<{name_w}}"
        for k in METRIC_KEYS:
            val = r.get(k)
            if val is None:
                row += f"  {'ERROR':>10}"
            elif isinstance(val, float) and not np.isnan(val):
                if k in ("Time_ms", "VRAM_MB"):
                    row += f"  {val:>10.1f}"
                else:
                    row += f"  {val:>10.4f}"
            else:
                row += f"  {'N/A':>10}"
        print(row)

    print("=" * len(header))


def save_csv(results, output_path):
    fieldnames = ["name"] + METRIC_KEYS + ["error"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nResults saved to {output_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def benchmark_one(model_info, weights_dir, test_data, device):
    """Benchmark a single model. Returns dict with metrics or error."""
    ckpt_path = os.path.join(weights_dir, model_info["ckpt"])
    if not os.path.isfile(ckpt_path):
        print(f"  SKIP: checkpoint not found at {ckpt_path}")
        return {"name": model_info["name"], "error": "checkpoint not found"}

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        model = build_opencd_model(model_info["config"], ckpt_path, device)
        preds, times_ms = run_opencd_inference(model, test_data, device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  CUDA OOM: {e}")
            torch.cuda.empty_cache()
            return {"name": model_info["name"], "error": "CUDA OOM"}
        raise
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"name": model_info["name"], "error": str(e)}

    peak_vram = 0.0
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gts = [item[2] for item in test_data]
    metrics = compute_metrics(preds, gts)

    avg_time = np.mean(times_ms) if times_ms else 0.0

    return {
        "name": model_info["name"],
        **metrics,
        "Time_ms": float(avg_time),
        "VRAM_MB": float(peak_vram),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark CD models on LEVIR-CD")
    parser.add_argument(
        "--data-root", default=None,
        help="Directory with A/, B/, label/ subdirs. Also used for weights unless --weights-dir is set.",
    )
    parser.add_argument(
        "--weights-dir", default=None,
        help="Directory with .pth/.ckpt files. Defaults to same as --data-root.",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Run only these model names (space-separated). Default: all.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: 'cuda', 'cpu', or 'auto' (default).",
    )
    parser.add_argument(
        "--output", default="benchmark_results.csv",
        help="Path for CSV output (default: benchmark_results.csv).",
    )
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("DATA_ROOT")
    weights_dir = args.weights_dir or os.environ.get("WEIGHTS_DIR") or data_root
    if not data_root:
        print("ERROR: Provide --data-root /path/to/LEVIR-CD/test")
        print("  Must contain A/, B/, label/ subdirs")
        print("  Weights (.pth/.ckpt) are expected in the same dir unless --weights-dir is set")
        sys.exit(1)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"Loading test data from {data_root} ...")
    test_data = load_test_data(data_root)
    print(f"Loaded {len(test_data)} image pairs\n")

    all_models = OPENCD_MODELS
    if args.models:
        selected = set(args.models)
        all_models = [m for m in all_models if m["name"] in selected]
        missing = selected - {m["name"] for m in all_models}
        if missing:
            print(f"Warning: unknown model names: {missing}")
            print(f"Available: {[m['name'] for m in OPENCD_MODELS]}")

    results = []
    for i, model_info in enumerate(all_models, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(all_models)}] {model_info['name']}")
        print(f"{'=' * 60}")

        result = benchmark_one(model_info, weights_dir, test_data, device)
        results.append(result)

        if "error" not in result:
            print(f"  F1={result['F1']:.4f}  IoU={result['IoU']:.4f}  "
                  f"BF1={result['BF1']:.4f}  Time={result['Time_ms']:.1f}ms  "
                  f"VRAM={result['VRAM_MB']:.0f}MB")

    successful = [r for r in results if "error" not in r]
    if successful:
        print_table(successful)
        save_csv(results, args.output)

        # Rank models by composite score (higher = better for accuracy, lower = better for time/vram)
        accuracy_keys = ["F1", "IoU", "Precision", "Recall", "BF1", "SmallRec"]
        for r in successful:
            valid_scores = [r[k] for k in accuracy_keys if isinstance(r.get(k), float) and not np.isnan(r[k])]
            r["AvgScore"] = np.mean(valid_scores) if valid_scores else 0.0

        ranked = sorted(successful, key=lambda r: r["AvgScore"], reverse=True)

        print("\n" + "=" * 60)
        print("  RANKING (by average of F1, IoU, Prec, Rec, BF1, SmallRec)")
        print("=" * 60)
        for i, r in enumerate(ranked, 1):
            print(f"  {i:>2}. {r['name']:<22}  AvgScore={r['AvgScore']:.4f}  "
                  f"F1={r['F1']:.4f}  Time={r['Time_ms']:.0f}ms  VRAM={r['VRAM_MB']:.0f}MB")

        print(f"\n  Best overall:   {ranked[0]['name']}")
        fastest = min(successful, key=lambda r: r["Time_ms"])
        lightest = min(successful, key=lambda r: r["VRAM_MB"])
        print(f"  Fastest:        {fastest['name']} ({fastest['Time_ms']:.0f}ms)")
        print(f"  Lowest VRAM:    {lightest['name']} ({lightest['VRAM_MB']:.0f}MB)")
    else:
        print("\nNo models completed successfully.")

    if [r for r in results if "error" in r]:
        print("\nFailed models:")
        for r in results:
            if "error" in r:
                print(f"  {r['name']}: {r['error']}")


if __name__ == "__main__":
    main()
