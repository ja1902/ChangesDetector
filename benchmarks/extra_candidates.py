#!/usr/bin/env python3
"""Second-round candidates: possible changes the first review never showed you.

The first round (candidates.py) only shows what three models flag at their usual
thresholds. This looks again with more sensitive thresholds and adds a model that
makes different mistakes: one fine-tuned on your other squares but never on this
one (benchmarks/loco_finetune.py). Only blobs outside every first-round candidate
are kept, so you review just the new ones.

Writes extra.gpkg per square (your candidates.gpkg is not touched):
  extra          the new candidates; review them like the first ones
                 (verdict real / false / unsure, type)
  missed_areas   empty polygon layer: draw changes no candidate covers, e.g. new
                 roads (no model here was trained on roads), with their type

    python benchmarks/extra_candidates.py --squares work_dirs/train_squares
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from osgeo import gdal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from candidates import save_prob, write_candidates  # noqa: E402
from score_testset import burn  # noqa: E402

gdal.UseExceptions()

# More sensitive than the first round (default 0.70, ensemble 0.35, fused 0.70).
THRESHOLDS = {"default": 0.4, "ensemble": 0.2, "fused": 0.5, "loco": 0.4}


def fold_model_for(square, loco_dir):
    """A fine-tuned model that never trained on this square."""
    own = loco_dir / square / "dinov2_finetuned.pth"
    if own.is_file():
        return own                        # trained on the other squares only
    models = sorted(loco_dir.glob("*/dinov2_finetuned.pth"))
    return models[0] if models else None  # trained on reviewed squares, not this one


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squares", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--loco", default="work_dirs/loco", help="folder of loco_finetune.py's fold models")
    ap.add_argument("--min-m2", type=float, default=50.0)
    ap.add_argument("--margin-m", type=float, default=3.0, help="keep this far from first-round candidates")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loco_dir = Path(args.loco)

    for sq in sorted(p for p in Path(args.squares).iterdir() if (p / "candidates.gpkg").is_file()):
        if args.only and sq.name not in args.only.split(","):
            continue
        if not (sq / "prob_loco.tif").is_file():
            m = fold_model_for(sq.name, loco_dir)
            if m is not None:
                from loco_finetune import infer
                prob, geo, _ = infer(m, sq, device)
                save_prob(sq / "prob_loco.tif", prob, geo)
                (sq / "prob_loco.txt").write_text(f"model: {m}\n")
        ds = gdal.Open(str(sq / "before.tif"))
        gt = ds.GetGeoTransform()
        px = abs(gt[1])
        geo = {"geotransform": gt, "projection": ds.GetProjection()}
        probs = {d: gdal.Open(str(sq / f"prob_{d}.tif")).ReadAsArray().astype(np.float32) / 255
                 for d in THRESHOLDS if (sq / f"prob_{d}.tif").is_file()}
        masks = {d: probs[d] >= THRESHOLDS[d] for d in probs}
        union = np.logical_or.reduce(list(masks.values()))
        covered = burn(ds, sq / "candidates.gpkg", "candidates", "1 = 1")[0] > 0
        covered |= burn(ds, sq / "candidates.gpkg", "small_ignored", "1 = 1")[0] > 0
        r = max(1, int(round(args.margin_m / px)))
        covered = cv2.dilate(covered.astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        new = (union & ~covered).astype(np.uint8)
        n, comp, st, _ = cv2.connectedComponentsWithStats(new, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] * px * px >= args.min_m2
        comp = np.where(keep[comp], comp, 0)
        idx = comp.ravel()
        stats = {int(i): {"sources": [], "p": {}, "area_m2": float(st[i, cv2.CC_STAT_AREA] * px * px)}
                 for i in np.flatnonzero(keep)}
        for d in probs:
            hit = np.bincount(idx, weights=masks[d].ravel().astype(np.float64), minlength=n)
            peak = np.zeros(n)
            np.maximum.at(peak, idx, probs[d].ravel())
            for i in stats:
                if hit[i] > 0:
                    stats[i]["sources"].append(d)
                stats[i]["p"][d] = float(peak[i])
        write_candidates(sq / "extra.gpkg", comp, stats, geo, layer="extra", second_round=True)
        by = {}
        for s in stats.values():
            for d in s["sources"]:
                by[d] = by.get(d, 0) + 1
        print(f"{sq.name}: {len(stats)} new candidates (found by: "
              + ", ".join(f"{d} {k}" for d, k in sorted(by.items())) + ")")


if __name__ == "__main__":
    main()
