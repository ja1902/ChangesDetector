#!/usr/bin/env python3
"""Object-level scores on the hand-checked test pairs.

Ground truth per pair (a folder made by prep_pairs.py + candidates.py):
  candidates.gpkg / candidates   polygons whose verdict is "real" are changes,
                                 "unsure" ones are ignored areas
  candidates.gpkg / missed       points: changes no candidate covered
or, for checking this script on a public dataset, label.tif (change mask).

For each detector (prob_<name>.tif at its preset threshold):
  recall          share of real changes covered by one of its blobs (points: within
                  --tol-m of a blob)
  false alarms    its blobs touching no real change and no "unsure" area, per km2
Also the union of all detectors: what the candidate review can find at best.

    python benchmarks/score_testset.py --pairs testset/
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from osgeo import gdal, ogr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

gdal.UseExceptions()
ogr.UseExceptions()


def read(path):
    ds = gdal.Open(str(path))
    return ds.GetRasterBand(1).ReadAsArray(), ds


def burn(ds_like, gpkg, layer, where):
    """Rasterise the matching polygons, one id per feature (0 = none)."""
    h, w = ds_like.RasterYSize, ds_like.RasterXSize
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Int32)
    mem.SetGeoTransform(ds_like.GetGeoTransform())
    mem.SetProjection(ds_like.GetProjection())
    src = ogr.Open(str(gpkg))
    lyr = src.GetLayerByName(layer)
    lyr.SetAttributeFilter(where)
    types = {}
    tmp = ogr.GetDriverByName("Memory").CreateDataSource("")
    tl = tmp.CreateLayer("t", lyr.GetSpatialRef(), ogr.wkbPolygon)
    tl.CreateField(ogr.FieldDefn("oid", ogr.OFTInteger))
    for i, f in enumerate(lyr, start=1):
        g = f.GetGeometryRef()
        if g is None:
            continue
        nf = ogr.Feature(tl.GetLayerDefn())
        nf.SetGeometry(g.Clone())
        nf.SetField("oid", i)
        tl.CreateFeature(nf)
        types[i] = (f.GetField("type") or "unspecified").strip().lower() if "type" in f.keys() else "unspecified"
    gdal.RasterizeLayer(mem, [1], tl, options=["ATTRIBUTE=oid", "ALL_TOUCHED=TRUE"])
    return mem.GetRasterBand(1).ReadAsArray(), types


def points(gpkg, gt):
    """(row, col, type) of the 'missed' points."""
    src = ogr.Open(str(gpkg))
    lyr = src.GetLayerByName("missed")
    out = []
    if lyr is None:
        return out
    for f in lyr:
        g = f.GetGeometryRef()
        if g is None:
            continue
        x, y = g.GetX(), g.GetY()
        col = (x - gt[0]) / gt[1]
        row = (y - gt[3]) / gt[5]
        out.append((int(row), int(col), (f.GetField("type") or "unspecified").strip().lower()))
    return out


def review_layers(ds_like, folder):
    """Everything the review says about a square, from candidates.gpkg and extra.gpkg.

    real        list of (id raster, {id: type}, origin): candidates and second-round
                candidates marked real ("candidate", "extra") and polygons drawn in
                missed_areas ("drawn"), which may sit in either file
    unsure      candidates of either round marked unsure
    unreviewed  candidates of either round with no verdict yet
    small       first-round candidates set aside for being small
    points      "missed" points (row, col, type)
    """
    shape = (ds_like.RasterYSize, ds_like.RasterXSize)
    out = {"real": [], "unsure": np.zeros(shape, bool), "unreviewed": np.zeros(shape, bool),
           "small": np.zeros(shape, bool), "points": []}
    for gpkg, layer, origin in ((folder / "candidates.gpkg", "candidates", "candidate"),
                                (folder / "extra.gpkg", "extra", "extra")):
        if not gpkg.is_file():
            continue
        out["real"].append((*burn(ds_like, gpkg, layer, "verdict = 'real'"), origin))
        out["unsure"] |= burn(ds_like, gpkg, layer, "verdict = 'unsure'")[0] > 0
        out["unreviewed"] |= burn(ds_like, gpkg, layer, "verdict IS NULL OR verdict = ''")[0] > 0
    src = ogr.Open(str(folder / "candidates.gpkg"))
    if src.GetLayerByName("small_ignored") is not None:
        out["small"] = burn(ds_like, folder / "candidates.gpkg", "small_ignored", "1 = 1")[0] > 0
    src = None
    for gpkg in (folder / "candidates.gpkg", folder / "extra.gpkg"):   # drawn changes, either file
        if gpkg.is_file():
            src = ogr.Open(str(gpkg))
            has = src.GetLayerByName("missed_areas") is not None
            src = None
            if has:
                out["real"].append((*burn(ds_like, gpkg, "missed_areas", "1 = 1"), "drawn"))
    out["points"] = points(folder / "candidates.gpkg", ds_like.GetGeoTransform())
    return out


def blobs(mask, min_px):
    n, comp, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px
    return np.where(keep[comp], comp, 0), np.flatnonzero(keep)


def score_pair(folder, dets, thr, tol_m, min_m2):
    first, ds = read(folder / f"prob_{dets[0]}.tif")
    gt = ds.GetGeoTransform()
    px = abs(gt[1])
    h, w = first.shape
    area_km2 = h * w * px * px / 1e6
    # Ground truth: object id map + types, ignore mask, points.
    if (folder / "label.tif").is_file():
        lab, _ = read(folder / "label.tif")
        n, obj = cv2.connectedComponents((lab > 127).astype(np.uint8), connectivity=8)
        types = {i: "change" for i in range(1, n)}
        ignore = np.zeros((h, w), bool)
        pts = []
    else:
        rv = review_layers(ds, folder)
        obj = np.zeros((h, w), np.int32)
        types = {}
        for ids, tps, _ in rv["real"]:
            for i, t in tps.items():
                k = len(types) + 1
                obj[(ids == i) & (obj == 0)] = k
                types[k] = t
        ignore = rv["unsure"] | rv["unreviewed"]
        pts = rv["points"]
    tol = max(1, int(round(tol_m / px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    min_px = max(1, int(round(min_m2 / (px * px))))

    res = {}
    masks = {}
    for d in dets:
        p, _ = read(folder / f"prob_{d}.tif")
        masks[d] = p.astype(np.float32) / 255 >= thr[d]
    masks["union"] = np.logical_or.reduce([masks[d] for d in dets])
    gt_any = (obj > 0) | ignore
    for d, mk in masks.items():
        comp, ids = blobs(mk, min_px)
        near = cv2.dilate((comp > 0).astype(np.uint8), kernel) > 0
        found = defaultdict(lambda: [0, 0])                     # type -> [found, total]
        hit_obj = np.unique(obj[(comp > 0) & (obj > 0)])
        for i, t in types.items():
            found[t][1] += 1
            found[t][0] += int(i in hit_obj)
        for r, c, t in pts:
            found[t][1] += 1
            found[t][0] += int(0 <= r < h and 0 <= c < w and near[r, c])
        # False alarm: a blob touching no real object, no unsure area and no missed point.
        touch = np.zeros(comp.max() + 1, bool)
        touch[np.unique(comp[gt_any & (comp > 0)])] = True
        for r, c, _ in pts:
            if 0 <= r < h and 0 <= c < w:
                touch[np.unique(comp[max(0, r - tol):r + tol + 1, max(0, c - tol):c + tol + 1])] = True
        fa = int(sum(1 for i in ids if not touch[i]))
        res[d] = {"found": dict(found), "false_alarms": fa, "blobs": len(ids)}
    return res, area_km2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--detectors", default="default,ensemble,fused")
    ap.add_argument("--tol-m", type=float, default=3.0, help="point-to-blob tolerance (m)")
    ap.add_argument("--min-m2", type=float, default=10.0, help="ignore detected blobs smaller than this")
    ap.add_argument("--csv", default=None, help="write per-pair results here")
    ap.add_argument("--only", default=None, help="comma list of pairs to score (default: all)")
    args = ap.parse_args()

    from bench import pipeline_args
    from candidates import DETECTORS, FUSED
    dets = args.detectors.split(",")
    thr = {d: pipeline_args(DETECTORS[FUSED.get(d, d)]).threshold for d in dets}

    folders = sorted(p for p in Path(args.pairs).iterdir() if (p / f"prob_{dets[0]}.tif").is_file())
    if args.only:
        folders = [p for p in folders if p.name in args.only.split(",")]
    tot = {d: {"found": defaultdict(lambda: [0, 0]), "fa": 0, "blobs": 0} for d in dets + ["union"]}
    km2 = 0.0
    rows = []
    for folder in folders:
        res, a = score_pair(folder, dets, thr, args.tol_m, args.min_m2)
        km2 += a
        for d, r in res.items():
            for t, (f, n) in r["found"].items():
                tot[d]["found"][t][0] += f
                tot[d]["found"][t][1] += n
            tot[d]["fa"] += r["false_alarms"]
            tot[d]["blobs"] += r["blobs"]
            f = sum(v[0] for v in r["found"].values())
            n = sum(v[1] for v in r["found"].values())
            rows.append({"pair": folder.name, "detector": d, "found": f, "changes": n,
                         "false_alarms": r["false_alarms"], "area_km2": round(a, 4)})

    n_obj = sum(v[1] for v in tot["union"]["found"].values())
    print(f"{len(folders)} pairs, {km2:.2f} km2, {n_obj} real changes\n")
    print(f"{'detector':10s} {'recall':>8s} {'found':>11s} {'false alarms/km2':>17s} {'blobs/km2':>10s}")
    for d, t in tot.items():
        f = sum(v[0] for v in t["found"].values())
        n = sum(v[1] for v in t["found"].values())
        print(f"{d:10s} {f / max(n, 1):8.3f} {f:5d}/{n:<5d} {t['fa'] / max(km2, 1e-9):17.1f} "
              f"{t['blobs'] / max(km2, 1e-9):10.1f}")
    types = sorted({k for t in tot.values() for k in t["found"]})
    if len(types) > 1:
        print("\nrecall by type: " + "  ".join(f"{k}" for k in types))
        for d, t in tot.items():
            print(f"{d:10s} " + "  ".join(f"{t['found'][k][0]}/{t['found'][k][1]}" for k in types))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
