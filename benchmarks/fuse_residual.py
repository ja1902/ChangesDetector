#!/usr/bin/env python3
"""Does the label-free change score (pair_residual.py) add anything to a trained model?

Fused probability: sigmoid(logit(p_model) + w * z), where z is the label-free score
(clu16_s on all four layers, in robust-z units, clipped). The weight w is chosen on
S2Looking *val* pairs; the fixed *test* pairs of the benchmark are only reported,
so the test number is not tuned on itself.

    python benchmarks/fuse_residual.py                           # shipped default model
    python benchmarks/fuse_residual.py --name fuse-ens -- --model-type ensemble --preset ensemble
Options after "--" go to detect_changes.py, as in bench.py.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import uchange_qgis_plugin.model_bridge  # noqa: E402,F401
from bench import Scores, data_root, load_set, read_rgb, pipeline_args, append_result, git_state  # noqa: E402
from pair_residual import PATCH, extract, clean_artifacts, score_cluster, robust_z  # noqa: E402

WEIGHTS = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
Z_CLIP = (-5.0, 20.0)


def label_free_z(bb, a, b, scale, device):
    """clu16_s on all layers, robust z, at the image resolution."""
    h, w = a.shape[:2]
    feats, (gh, gw), (hs, ws) = extract(bb, a, b, scale, 37, device)
    zs = []
    for f in feats:
        clean_artifacts(f)
        fa = f[0].reshape(f.shape[1], -1).T
        fb = f[1].reshape(f.shape[1], -1).T
        zs.append(robust_z(score_cluster(fa, fb, (gh, gw), True, k=16)))
    z = torch.stack(zs).mean(0).reshape(1, 1, gh, gw)
    z = F.interpolate(z, size=(gh * PATCH, gw * PATCH), mode="bilinear", align_corners=False)[:, :, :hs, :ws]
    z = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
    return z[0, 0].clamp(*Z_CLIP).cpu().numpy()


def pairs(root, split, n, seed):
    base = root / "S2Looking" / split
    if split == "test":
        stems = load_set("s2looking")
    else:
        stems = sorted(p.stem for p in (base / "label").glob("*.png"))
        random.Random(seed).shuffle(stems)
        stems = sorted(stems[:n])
    return [(base / "Image1" / f"{s}.png", base / "Image2" / f"{s}.png", base / "label" / f"{s}.png")
            for s in stems]


def run_split(name, plist, p, model, cs_model, bb, device):
    import detect_changes as dc
    scores = [Scores() for _ in WEIGHTS]
    t0 = time.time()
    for i, (pa, pb, pl) in enumerate(plist):
        a, b = read_rgb(pa), read_rgb(pb)
        lab = np.array(Image.open(pl).convert("L")) > 127
        prob = dc.infer_probability(p, model, cs_model, a, b, p.scale, device)
        z = label_free_z(bb, a, b, max(p.scale, 1.0), device)
        logit = np.log(np.clip(prob, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - prob, 1e-6, 1))
        for w, sc in zip(WEIGHTS, scores):
            f = 1 / (1 + np.exp(-(logit + w * z)))
            sc.add(f, f >= p.threshold, lab)
        print(f"\r{name}: {i + 1}/{len(plist)}  ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    return [sc.summary(False) for sc in scores]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="fuse-default")
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    extra = [x for x in args.extra if x != "--"]

    from uchange_qgis_plugin.model_bridge import build_model
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = data_root(args.data)
    p = pipeline_args(extra)
    weights = p.weights if os.path.isabs(p.weights) else str(ROOT / p.weights)
    model, _ = build_model(weights, device, model_type="dinov2" if p.model_type in ("ensemble", "dinov2_lc")
                           else p.model_type)
    cs_model = None
    if p.model_type == "ensemble":
        from uchange_qgis_plugin.changestar_bridge import build_changestar
        cs_model = build_changestar(device)[0]
    bb, _ = build_model(str(ROOT / "dinov2_vitb14_c2s1_levir.pth"), device, model_type="dinov2")

    val = run_split("val", pairs(root, "val", args.n_val, args.seed), p, model, cs_model, bb, device)
    k = int(np.argmax([s["best_iou"] for s in val]))
    test = run_split("test", pairs(root, "test", 0, 0), p, model, cs_model, bb, device)

    print(f"\n{'w':>6}   val best IoU / AP     test best IoU / AP   (model: {' '.join(extra) or 'default'})")
    for i, w in enumerate(WEIGHTS):
        mark = "  <- chosen on val" if i == k else ""
        print(f"{w:6.2f}   {val[i]['best_iou']:.3f} / {val[i]['ap']:.3f}        "
              f"{test[i]['best_iou']:.3f} / {test[i]['ap']:.3f}{mark}")
    print(f"\ntest: model alone {test[0]['best_iou']:.3f} / {test[0]['ap']:.3f}  ->  "
          f"fused (w={WEIGHTS[k]}) {test[k]['best_iou']:.3f} / {test[k]['ap']:.3f}")
    append_result({"kind": "fuse_residual", "name": args.name, "options": extra, "device": device.type,
                   "time": time.strftime("%Y-%m-%d %H:%M"), **git_state(), "weights": WEIGHTS,
                   "chosen_w": WEIGHTS[k], "val": val, "test": test})


if __name__ == "__main__":
    main()
