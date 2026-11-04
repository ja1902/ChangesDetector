#!/usr/bin/env python3
"""Change candidates for labelling a test pair by review instead of from scratch.

Runs several different detectors on each prepared pair (prep_pairs.py), merges
their change blobs and writes, per pair:

  candidates.gpkg   layer "candidates": one polygon per merged blob, with the
                    detectors that found it and their peak probabilities, and empty
                    fields for you: verdict (real / false / unsure), type, note
                    layer "missed": empty point layer for changes every detector missed
  prob_<detector>.tif   each detector's change probability (0-255), for scoring later

Detectors: "default" (the shipped DINOv2 model), "ensemble" (DINOv2 + Changen2
ChangeStar) and "fused" (default nudged by the label-free score of
pair_residual.py, weight 1.0 as chosen on S2Looking val). Several different models
keep the review from only confirming what one model finds; still scan each pair for
misses, or recall will look better than it is.

    python benchmarks/candidates.py --pairs testset/
    python benchmarks/candidates.py --pairs testset/ --only site_a --detectors default,fused
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from osgeo import gdal, ogr, osr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import uchange_qgis_plugin.model_bridge  # noqa: E402,F401
from bench import pipeline_args  # noqa: E402
from fuse_residual import label_free_z  # noqa: E402

gdal.UseExceptions()
ogr.UseExceptions()

DETECTORS = {"default": [], "ensemble": ["--model-type", "ensemble", "--preset", "ensemble"],
             "ft": ["--weights", "dinov2_vitb14_finetuned.pth", "--preset", "synthetic"]}   # fine-tuned on your squares, as the plugin runs it (nudge off)
FUSED = {"fused": "default", "ftz": "ft"}      # detector -> model it nudges with the label-free score
FUSE_W = 1.0
TYPES = "building / road / car_park / construction_site / demolition / other"


def big_label_free_z(bb, a, b, factor, device, block=1024, overlap=128):
    """label_free_z in overlapping blocks, the size it was tuned at (S2Looking, 1024 px)."""
    h, w = a.shape[:2]
    if max(h, w) <= block:
        return label_free_z(bb, a, b, factor, device)

    def starts(n):
        if n <= block:
            return [0]
        return sorted(set(list(range(0, n - block + 1, block - overlap)) + [n - block]))

    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for y in starts(h):
        for x in starts(w):
            sl = (slice(y, y + block), slice(x, x + block))
            acc[sl] += label_free_z(bb, a[sl], b[sl], factor, device)
            cnt[sl] += 1
    return acc / np.maximum(cnt, 1)


def save_prob(path, prob, geo):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), prob.shape[1], prob.shape[0], 1, gdal.GDT_Byte,
                    options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(geo["geotransform"])
    ds.SetProjection(geo["projection"])
    ds.GetRasterBand(1).WriteArray(np.clip(prob * 255 + 0.5, 0, 255).astype(np.uint8))
    ds = None


def write_candidates(path, comp, stats, geo, layer="candidates", second_round=False):
    """Polygonise the merged blob ids and attach per-blob attributes.

    Also adds an empty point layer "missed" (first round, candidates.gpkg) or an
    empty polygon layer "missed_areas" (second round, extra.gpkg) for changes that
    no candidate covers.
    """
    if Path(path).exists():
        os.remove(path)
    h, w = comp.shape
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Int32)
    mem.SetGeoTransform(geo["geotransform"])
    mem.SetProjection(geo["projection"])
    band = mem.GetRasterBand(1)
    band.WriteArray(comp.astype(np.int32))
    mask = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mask.GetRasterBand(1).WriteArray((comp > 0).astype(np.uint8))

    srs = osr.SpatialReference()
    srs.ImportFromWkt(geo["projection"])
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    lyr = ds.CreateLayer(layer, srs, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("cid", ogr.OFTInteger))
    gdal.Polygonize(band, mask.GetRasterBand(1), lyr, 0, ["8CONNECTED=8"])
    det_names = sorted({d for s in stats.values() for d in s["p"]})
    for name, kind in ([("sources", ogr.OFTString), ("n_sources", ogr.OFTInteger), ("area_m2", ogr.OFTReal)]
                       + [(f"p_{d}", ogr.OFTReal) for d in det_names]
                       + [("verdict", ogr.OFTString), ("type", ogr.OFTString), ("note", ogr.OFTString)]):
        lyr.CreateField(ogr.FieldDefn(name, kind))
    lyr.ResetReading()
    ds.StartTransaction()
    for f in lyr:
        s = stats[f.GetField("cid")]
        f.SetField("sources", "+".join(s["sources"]))
        f.SetField("n_sources", len(s["sources"]))
        f.SetField("area_m2", round(s["area_m2"], 1))
        for d, p in s["p"].items():
            f.SetField(f"p_{d}", round(p, 3))
        lyr.SetFeature(f)
    ds.CommitTransaction()

    from review_style import add_styles, candidates_qml, missed_areas_qml
    if second_round:
        areas = ds.CreateLayer("missed_areas", srs, ogr.wkbPolygon)
        for name in ("type", "direction", "note"):
            areas.CreateField(ogr.FieldDefn(name, ogr.OFTString))
        ds = None
        add_styles(path, {layer: candidates_qml(), "missed_areas": missed_areas_qml()})
    else:
        pts = ds.CreateLayer("missed", srs, ogr.wkbPoint)
        for name in ("type", "direction", "confidence", "note"):
            pts.CreateField(ogr.FieldDefn(name, ogr.OFTString))
        ds = None
        add_styles(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="output folder of prep_pairs.py")
    ap.add_argument("--only", default=None, help="comma list of pair names")
    ap.add_argument("--detectors", default="default,ensemble,fused")
    ap.add_argument("--min-m2", type=float, default=10.0, help="drop merged blobs smaller than this")
    args = ap.parse_args()

    import detect_changes as dc
    from uchange_qgis_plugin.model_bridge import build_model
    from uchange_qgis_plugin.resolution import working_factor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dets = args.detectors.split(",")
    cfg, models = {}, {}
    for d in dets:
        p = pipeline_args(DETECTORS[FUSED.get(d, d)])
        cfg[d] = p
    bases = [x for x in dets if x not in FUSED]
    bases += [FUSED[x] for x in dets if x in FUSED and FUSED[x] not in bases]
    for d in bases:
        p = cfg.get(d) or pipeline_args(DETECTORS[d])
        weights = p.weights if os.path.isabs(p.weights) else str(ROOT / p.weights)
        m, _ = build_model(weights, device, model_type="dinov2")
        cs = None
        if p.model_type == "ensemble":
            from uchange_qgis_plugin.changestar_bridge import build_changestar
            cs = build_changestar(device)[0]
        models[d] = (p, m, cs)
    bb = (build_model(str(ROOT / "dinov2_vitb14_c2s1_levir.pth"), device, model_type="dinov2")[0]
          if any(d in FUSED for d in dets) else None)

    folders = sorted(p for p in Path(args.pairs).iterdir() if (p / "before.tif").is_file())
    if args.only:
        folders = [p for p in folders if p.name in args.only.split(",")]
    for folder in folders:
        t0 = time.time()
        a, geo = dc.read_image(str(folder / "before.tif"))
        b, _ = dc.read_image(str(folder / "after.tif"))
        gt = geo["geotransform"]
        px_m2 = abs(gt[1] * gt[5])
        probs = {}
        for d in bases:
            p, m, cs = models[d]
            factor, _, _ = working_factor(geo, p.target_gsd, p.scale, a.shape)
            probs[d] = dc.infer_probability(p, m, cs, a, b, factor, device)
        fused = [d for d in dets if d in FUSED]
        if fused:
            p = cfg[fused[0]]
            factor, _, _ = working_factor(geo, p.target_gsd, p.scale, a.shape)
            z = big_label_free_z(bb, a, b, factor, device)
            for d in fused:
                pd = np.clip(probs[FUSED[d]], 1e-6, 1 - 1e-6)
                probs[d] = 1 / (1 + np.exp(-(np.log(pd / (1 - pd)) + FUSE_W * z)))
        probs = {d: probs[d] for d in dets}
        for d, pr in probs.items():
            save_prob(folder / f"prob_{d}.tif", pr, geo)

        masks = {d: probs[d] >= cfg[d].threshold for d in dets}
        union = np.zeros(a.shape[:2], np.uint8)
        for mk in masks.values():
            union |= mk.astype(np.uint8)
        n, comp, cc_stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = cc_stats[1:, cv2.CC_STAT_AREA] * px_m2 >= args.min_m2
        comp = np.where(keep[comp], comp, 0)
        stats = {}
        idx = comp.ravel()
        for i in np.flatnonzero(keep):
            stats[int(i)] = {"sources": [], "p": {}, "area_m2": float(cc_stats[i, cv2.CC_STAT_AREA] * px_m2)}
        for d in dets:
            hit = np.bincount(idx, weights=masks[d].ravel().astype(np.float64), minlength=n)
            peak = np.zeros(n)
            np.maximum.at(peak, idx, probs[d].ravel())
            for i in stats:
                if hit[i] > 0:
                    stats[i]["sources"].append(d)
                stats[i]["p"][d] = float(peak[i])
        write_candidates(folder / "candidates.gpkg", comp, stats, geo)
        by = {k: sum(1 for s in stats.values() if len(s["sources"]) == k) for k in range(1, len(dets) + 1)}
        area_km2 = a.shape[0] * a.shape[1] * px_m2 / 1e6
        print(f"{folder.name}: {len(stats)} candidates on {area_km2:.2f} km2 "
              f"(found by 1/2/3 detectors: {'/'.join(str(by.get(k, 0)) for k in (1, 2, 3))})  "
              f"{time.time() - t0:.0f}s")
    print(f"\nReview in QGIS: open before.tif, after.tif and candidates.gpkg; set verdict "
          f"(real / false / unsure) and type ({TYPES}); add a point to 'missed' for changes "
          f"no candidate covers.")


if __name__ == "__main__":
    main()
