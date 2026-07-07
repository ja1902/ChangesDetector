"""
Visualize ChangerEx_r18 predictions on a LEVIR-CD test image.

Produces a 2x2 panel:
  Before | After
  Prediction overlay (red on before) | TP/FP/FN comparison

Usage:
    python visualize_cd.py \
        --data-root /path/to/LEVIR-CD/test \
        --weights-dir /path/to/weights
    python visualize_cd.py --data-root ... --weights-dir ... --image 5
    python visualize_cd.py --data-root ... --weights-dir ... --image test_3.png
"""

import os
import sys
import argparse
import warnings

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIG = "configs/changer/changer_ex_r18_512x512_40k_levircd.py"
MODEL_CKPT = "ChangerEx_r18-512x512_40k_levircd.pth"


def build_model(ckpt_path, device):
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import DefaultScope
    import opencd.models  # noqa: F401

    cfg = Config.fromfile(os.path.join(PROJECT_ROOT, MODEL_CONFIG))
    if hasattr(cfg.model, "pretrained"):
        cfg.model.pretrained = None
    for attr in ("backbone", "image_encoder", "decode_head", "auxiliary_head"):
        sub = getattr(cfg.model, attr, None)
        if sub is not None and hasattr(sub, "init_cfg"):
            sub.init_cfg = None

    from opencd.registry import MODELS
    DefaultScope.get_instance('opencd', scope_name='opencd')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location="cpu", logger=None)
    model = revert_sync_batchnorm(model)
    model.to(device).eval()
    return model


def run_inference(model, img_a_bgr, img_b_bgr, device):
    from mmseg.structures import SegDataSample

    a_chw = torch.from_numpy(img_a_bgr.transpose(2, 0, 1).copy())
    b_chw = torch.from_numpy(img_b_bgr.transpose(2, 0, 1).copy())
    tensor_6ch = torch.cat([a_chw, b_chw], dim=0)

    h, w = img_a_bgr.shape[:2]
    data_sample = SegDataSample()
    data_sample.set_metainfo({
        "ori_shape": (h, w),
        "img_shape": (h, w),
        "pad_shape": (h, w),
        "padding_size": [0, 0, 0, 0],
    })

    data = {"inputs": [tensor_6ch], "data_samples": [data_sample]}

    with torch.no_grad():
        results = model.test_step(data)

    return results[0].pred_sem_seg.data.cpu().numpy().squeeze().astype(np.uint8)


def make_overlay(img_rgb, mask, color=(255, 0, 0), alpha=0.4):
    overlay = img_rgb.copy()
    overlay[mask > 0] = (
        (1 - alpha) * overlay[mask > 0] + alpha * np.array(color)
    ).astype(np.uint8)
    return overlay


def make_comparison(pred, gt):
    h, w = gt.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    tp = (pred > 0) & (gt > 0)
    fp = (pred > 0) & (gt == 0)
    fn = (pred == 0) & (gt > 0)
    vis[tp] = [0, 200, 0]    # green
    vis[fp] = [220, 40, 40]  # red
    vis[fn] = [50, 80, 220]  # blue
    return vis


def main():
    parser = argparse.ArgumentParser(description="Visualize ChangerEx_r18 CD predictions")
    parser.add_argument("--data-root", required=True, help="LEVIR-CD test dir with A/, B/, label/")
    parser.add_argument("--weights-dir", required=True, help="Directory with .pth files")
    parser.add_argument("--image", default="0", help="Image filename or index (default: 0)")
    parser.add_argument("--output", default="visualization.png", help="Output PNG path")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    a_dir = os.path.join(args.data_root, "A")
    b_dir = os.path.join(args.data_root, "B")
    label_dir = os.path.join(args.data_root, "label")

    fnames = sorted(f for f in os.listdir(a_dir) if f.lower().endswith((".png", ".jpg", ".tif")))

    if args.image.isdigit():
        idx = int(args.image)
        fname = fnames[idx]
    else:
        fname = args.image

    print(f"Image: {fname}")

    img_a_bgr = cv2.imread(os.path.join(a_dir, fname))
    img_b_bgr = cv2.imread(os.path.join(b_dir, fname))
    gt = cv2.imread(os.path.join(label_dir, fname), cv2.IMREAD_GRAYSCALE)
    gt_bin = (gt >= 128).astype(np.uint8)

    img_a_rgb = img_a_bgr[:, :, ::-1].copy()
    img_b_rgb = img_b_bgr[:, :, ::-1].copy()

    ckpt_path = os.path.join(args.weights_dir, MODEL_CKPT)
    print(f"Loading model...")
    model = build_model(ckpt_path, device)

    print(f"Running inference...")
    pred = run_inference(model, img_a_bgr, img_b_bgr, device)

    overlay = make_overlay(img_a_rgb, pred, color=(255, 30, 30), alpha=0.45)
    comparison = make_comparison(pred, gt_bin)

    tp = ((pred > 0) & (gt_bin > 0)).sum()
    fp = ((pred > 0) & (gt_bin == 0)).sum()
    fn = ((pred == 0) & (gt_bin > 0)).sum()
    eps = 1e-8
    f1 = 2 * tp / (2 * tp + fp + fn + eps)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    axes[0, 0].imshow(img_a_rgb)
    axes[0, 0].set_title("Before (A)", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_b_rgb)
    axes[0, 1].set_title("After (B)", fontsize=14)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Predicted changes (red overlay)", fontsize=14)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(comparison)
    axes[1, 1].set_title(f"GT vs Pred — F1={f1:.4f}", fontsize=14)
    axes[1, 1].axis("off")

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=(0, 0.78, 0), label="TP (correct detection)"),
        Patch(facecolor=(0.86, 0.16, 0.16), label="FP (false alarm)"),
        Patch(facecolor=(0.20, 0.31, 0.86), label="FN (missed change)"),
    ]
    axes[1, 1].legend(handles=legend, loc="lower right", fontsize=10,
                      framealpha=0.8, edgecolor="gray")

    fig.suptitle(f"ChangerEx_r18 — {fname}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
