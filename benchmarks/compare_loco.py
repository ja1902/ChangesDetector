#!/usr/bin/env python3
"""Compare techniques on the held-out squares (loco_finetune.py runs), threshold-free.

A single threshold favours whichever model happens to be calibrated for it, so each
model is swept over thresholds and compared at equal footing:
  AP / best IoU     pixels, against train_label.tif (ignore areas left out), mean over squares
  recall @ N FA     object recall when false alarms are held to N per km2 (pooled)
  FA @ R recall     false alarms per km2 needed to reach recall R (pooled)
  best thr          threshold giving recall @ N FA

    python benchmarks/compare_loco.py --squares work_dirs/train_squares --names default,ft_base,ft_hn
    ... --names ft_base,ft_base_z0.5,ft_base_z1.0     # NAME_zW: NAME nudged by the label-free
                                                      # score (z.tif, save_zmaps.py) with weight W
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from loco_finetune import pixel_scores  # noqa: E402
from candidates import save_prob  # noqa: E402
from save_zmaps import nudge  # noqa: E402
from score_testset import score_pair  # noqa: E402

THRESHOLDS = np.round(np.concatenate([np.arange(0.2, 0.9, 0.05), np.arange(0.9, 0.995, 0.01)]), 2)


def curve(squares, name, min_m2=50.0, tol_m=3.0):
    """Pooled (threshold, recall, false alarms per km2) over the squares."""
    rows = []
    for t in THRESHOLDS:
        f = n = fa = km2 = 0
        for sq in squares:
            res, a = score_pair(sq, [name], {name: float(t)}, tol_m, min_m2)
            r = res[name]
            f += sum(v[0] for v in r["found"].values())
            n += sum(v[1] for v in r["found"].values())
            fa += r["false_alarms"]
            km2 += a
        rows.append((float(t), f / max(n, 1), fa / km2))
    return rows


def at_fa(rows, budget):
    ok = [(rec, t) for t, rec, fa in rows if fa <= budget]
    return max(ok) if ok else (0.0, None)


def fa_at(rows, recall):
    ok = [fa for t, rec, fa in rows if rec >= recall]
    return min(ok) if ok else float("inf")


def make_nudged(folder, name):
    """prob_NAME_zW.tif from prob_NAME.tif and z.tif, if NAME_zW asks for it."""
    m = re.fullmatch(r"(.+)_z([\d.]+)", name)
    src, out = folder / f"prob_{m.group(1)}.tif" if m else None, folder / f"prob_{name}.tif"
    if not m or not src.is_file() or not (folder / "z.tif").is_file():
        return
    if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
        return
    from osgeo import gdal
    ds = gdal.Open(str(src))
    p = ds.ReadAsArray().astype(np.float32) / 255
    z = gdal.Open(str(folder / "z.tif")).ReadAsArray()
    save_prob(out, nudge(p, z, float(m.group(2))),
              {"geotransform": ds.GetGeoTransform(), "projection": ds.GetProjection()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squares", required=True)
    ap.add_argument("--names", default="default,ft_base")
    ap.add_argument("--fa", type=float, default=20.0, help="false-alarm budget per km2")
    ap.add_argument("--recall", type=float, default=0.85)
    ap.add_argument("--only", default=None, help="comma list of squares (default: every labelled one)")
    args = ap.parse_args()
    names = args.names.split(",")
    folders = sorted(p for p in Path(args.squares).iterdir() if (p / "train_label.tif").is_file())
    if args.only:
        folders = [p for p in folders if p.name in args.only.split(",")]
    print(f"{'model':12s} {'squares':>7s} {'AP':>6s} {'bestIoU':>7s} "
          f"{'recall@' + str(int(args.fa)) + 'FA':>12s} {'thr':>5s} {'FA@' + str(args.recall):>9s}")
    for nm in names:
        for p in folders:
            make_nudged(p, nm)
        sq = [p for p in folders if (p / f"prob_{nm}.tif").is_file()]
        if not sq:
            print(f"{nm:12s} (no prob_{nm}.tif yet)")
            continue
        pix = [pixel_scores(p, [nm])[nm] for p in sq]
        rows = curve(sq, nm)
        rec, thr = at_fa(rows, args.fa)
        print(f"{nm:12s} {len(sq):7d} {np.mean([p['ap'] for p in pix]):6.3f} "
              f"{np.mean([p['best_iou'] for p in pix]):7.3f} {rec:12.3f} "
              f"{thr if thr is not None else '-':>5} {fa_at(rows, args.recall):9.1f}", flush=True)
        per = "  ".join(f"{p.name.split('_')[0]} {x['ap']:.2f}" for p, x in zip(sq, pix))
        print(f"{'':12s} AP per square: {per}")


if __name__ == "__main__":
    main()
