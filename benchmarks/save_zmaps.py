#!/usr/bin/env python3
"""Save each square's label-free change score (z.tif, float32) for nudging any model later.

z is the per-pair residual score of benchmarks/fuse_residual.py (clu16_s, all layers),
as candidates.py computes it for the "fused" detector. A model's probability p is
nudged as sigmoid(logit(p) + w * z): see nudge().

    python benchmarks/save_zmaps.py --squares work_dirs/train_squares
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from osgeo import gdal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import uchange_qgis_plugin.model_bridge  # noqa: E402,F401
from candidates import big_label_free_z  # noqa: E402

gdal.UseExceptions()


def nudge(p, z, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + w * z)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squares", required=True)
    args = ap.parse_args()
    import detect_changes as dc
    from bench import pipeline_args
    from uchange_qgis_plugin.model_bridge import build_model
    from uchange_qgis_plugin.resolution import working_factor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = build_model(str(ROOT / "dinov2_vitb14_c2s1_levir.pth"), device, model_type="dinov2")[0]
    p = pipeline_args([])
    for sq in sorted(q for q in Path(args.squares).iterdir() if (q / "before.tif").is_file()):
        if (sq / "z.tif").is_file():
            continue
        a, geo = dc.read_image(str(sq / "before.tif"))
        b, _ = dc.read_image(str(sq / "after.tif"))
        factor, _, _ = working_factor(geo, p.target_gsd, p.scale, a.shape)
        z = big_label_free_z(bb, a, b, factor, device).astype(np.float32)
        ds = gdal.GetDriverByName("GTiff").Create(str(sq / "z.tif"), z.shape[1], z.shape[0], 1,
                                                  gdal.GDT_Float32, options=["COMPRESS=DEFLATE"])
        ds.SetGeoTransform(geo["geotransform"])
        ds.SetProjection(geo["projection"])
        ds.GetRasterBand(1).WriteArray(z)
        ds = None
        print(f"{sq.name}: z range {z.min():.1f}..{z.max():.1f}", flush=True)


if __name__ == "__main__":
    main()
