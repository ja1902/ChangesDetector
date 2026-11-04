#!/usr/bin/env python3
"""One review file per square: both rounds of candidates, ready for QGIS.

Uses the probability maps candidates.py wrote (prob_<detector>.tif) and replaces its
candidates.gpkg with:
  candidates     round 1: blobs any detector flags at its usual threshold;
                 round 2: blobs flagged only at the sensitive thresholds, away from
                 round 1 (field "round"). Review both alike: verdict and type
  small_ignored  round-1 blobs under --min-m2 (not reviewed; ignored in training)
  missed_areas   empty: draw changes no candidate covers (roads, sites), with a type
plus grid.gpkg (4 x 4 cells with a "done" field) to track progress.

    python benchmarks/candidates.py --pairs work_dirs/new_squares --detectors default,ensemble,fused,ft,ftz
    python benchmarks/review_pack.py --squares work_dirs/new_squares
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from osgeo import gdal, ogr, osr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from review_style import add_styles, candidates_qml, make_grid, missed_areas_qml  # noqa: E402

gdal.UseExceptions()
ogr.UseExceptions()

ROUND1 = {"default": 0.70, "ensemble": 0.35, "fused": 0.70, "ft": 0.70, "ftz": 0.85}
ROUND2 = {"default": 0.40, "ensemble": 0.20, "fused": 0.50, "ft": 0.40, "ftz": 0.60}


def blob_stats(comp, keep, n, probs, masks, px_m2, rnd):
    idx = comp.ravel()
    area = np.bincount(idx, minlength=n)
    stats = {int(i): {"round": rnd, "sources": [], "p": {}, "area_m2": float(area[i] * px_m2)}
             for i in np.flatnonzero(keep)}
    for d in probs:
        hit = np.bincount(idx, weights=masks[d].ravel().astype(np.float64), minlength=n)
        peak = np.zeros(n)
        np.maximum.at(peak, idx, probs[d].ravel())
        for i in stats:
            if hit[i] > 0:
                stats[i]["sources"].append(d)
            stats[i]["p"][d] = float(peak[i])
    return stats


def polygonize(ds, name, ids, geo, srs):
    h, w = ids.shape
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Int32)
    mem.SetGeoTransform(geo.GetGeoTransform())
    mem.SetProjection(geo.GetProjection())
    mem.GetRasterBand(1).WriteArray(ids.astype(np.int32))
    msk = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    msk.GetRasterBand(1).WriteArray((ids > 0).astype(np.uint8))
    lyr = ds.CreateLayer(name, srs, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("cid", ogr.OFTInteger))
    gdal.Polygonize(mem.GetRasterBand(1), msk.GetRasterBand(1), lyr, 0, ["8CONNECTED=8"])
    return lyr


def pack(sq, min_m2, margin_m, tiny_m2):
    geo = gdal.Open(str(sq / "before.tif"))
    gt = geo.GetGeoTransform()
    px = abs(gt[1])
    px_m2 = px * px
    dets = [d for d in ROUND1 if (sq / f"prob_{d}.tif").is_file()]
    probs = {d: gdal.Open(str(sq / f"prob_{d}.tif")).ReadAsArray().astype(np.float32) / 255 for d in dets}

    # Round 1: usual thresholds. Blobs under min_m2 go to small_ignored.
    m1 = {d: probs[d] >= ROUND1[d] for d in dets}
    u1 = np.logical_or.reduce(list(m1.values())).astype(np.uint8)
    n1, c1, st1, _ = cv2.connectedComponentsWithStats(u1, connectivity=8)
    a1 = st1[:, cv2.CC_STAT_AREA] * px_m2
    big = np.zeros(n1, bool)
    big[1:] = a1[1:] >= min_m2
    small = np.zeros(n1, bool)
    small[1:] = (a1[1:] < min_m2) & (a1[1:] >= tiny_m2)
    s1 = blob_stats(np.where(big[c1], c1, 0), big, n1, probs, m1, px_m2, 1)

    # Round 2: sensitive thresholds, only what lies away from round 1 (and its small blobs).
    m2 = {d: probs[d] >= ROUND2[d] for d in dets}
    u2 = np.logical_or.reduce(list(m2.values()))
    r = max(1, int(round(margin_m / px)))
    near1 = cv2.dilate((c1 > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    n2, c2, st2, _ = cv2.connectedComponentsWithStats((u2 & ~near1).astype(np.uint8), connectivity=8)
    big2 = np.zeros(n2, bool)
    big2[1:] = st2[1:, cv2.CC_STAT_AREA] * px_m2 >= min_m2
    s2 = blob_stats(np.where(big2[c2], c2, 0), big2, n2, probs, m2, px_m2, 2)

    # One id raster for both rounds (round 2 ids shifted past round 1's).
    ids = np.where(big[c1], c1, 0).astype(np.int32)
    off = n1
    ids = np.where((ids == 0) & big2[c2], c2 + off, ids)
    stats = dict(s1)
    stats.update({i + off: s for i, s in s2.items()})

    path = sq / "candidates.gpkg"
    for ext in ("", "-wal", "-shm"):
        if Path(str(path) + ext).exists():
            os.remove(str(path) + ext)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(geo.GetProjection())
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    lyr = polygonize(ds, "candidates", ids, geo, srs)
    for name, kind in ([("round", ogr.OFTInteger), ("sources", ogr.OFTString), ("n_sources", ogr.OFTInteger),
                        ("area_m2", ogr.OFTReal)] + [(f"p_{d}", ogr.OFTReal) for d in dets]
                       + [("verdict", ogr.OFTString), ("type", ogr.OFTString), ("note", ogr.OFTString)]):
        lyr.CreateField(ogr.FieldDefn(name, kind))
    lyr.ResetReading()
    ds.StartTransaction()
    for f in lyr:
        s = stats[f.GetField("cid")]
        f.SetField("round", s["round"])
        f.SetField("sources", "+".join(s["sources"]))
        f.SetField("n_sources", len(s["sources"]))
        f.SetField("area_m2", round(s["area_m2"], 1))
        for d, p in s["p"].items():
            f.SetField(f"p_{d}", round(p, 3))
        lyr.SetFeature(f)
    ds.CommitTransaction()
    sm = polygonize(ds, "small_ignored", np.where(small[c1], c1, 0), geo, srs)
    sm.CreateField(ogr.FieldDefn("area_m2", ogr.OFTReal))
    sm.ResetReading()
    ds.StartTransaction()
    for f in sm:
        f.SetField("area_m2", round(float(a1[f.GetField("cid")]), 1))
        sm.SetFeature(f)
    ds.CommitTransaction()
    areas = ds.CreateLayer("missed_areas", srs, ogr.wkbPolygon)
    for name in ("type", "direction", "note"):
        areas.CreateField(ogr.FieldDefn(name, ogr.OFTString))
    ds = None
    add_styles(path, {"candidates": candidates_qml(), "missed_areas": missed_areas_qml()})
    make_grid(sq)
    return len(s1), len(s2), int(small.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squares", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--min-m2", type=float, default=50.0, help="smaller round-1 blobs go to small_ignored")
    ap.add_argument("--tiny-m2", type=float, default=10.0, help="blobs under this are dropped altogether")
    ap.add_argument("--margin-m", type=float, default=3.0, help="round 2 keeps this far from round 1")
    args = ap.parse_args()
    for sq in sorted(p for p in Path(args.squares).iterdir() if (p / "before.tif").is_file()):
        if args.only and sq.name not in args.only.split(","):
            continue
        n1, n2, ns = pack(sq, args.min_m2, args.margin_m, args.tiny_m2)
        print(f"{sq.name}: {n1} + {n2} (round 2) candidates to review, {ns} small set aside", flush=True)


if __name__ == "__main__":
    main()
