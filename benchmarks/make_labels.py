#!/usr/bin/env python3
"""Training labels from reviewed candidates (candidates.py + a review in QGIS).

Per square, writes train_label.tif: 0 = no change, 1 = change, 255 = ignore (not trained on):
  change     candidates marked "real" (either round), outlines tightened with Segment
             Anything 2 (on the after image, or the before image for demolitions) when
             SAM's outline agrees with the candidate; otherwise the candidate as it is.
             Polygons drawn in missed_areas: kept as drawn, except buildings up to
             3000 m2, which SAM tightens. A drawing overrides "false" candidates under it
  ignore     a band around every change outline (wider if SAM was not used), candidates
             marked "unsure" or left unreviewed, the small candidates set aside
             (small_ignored) and a disc around each "missed" point (no outline for it)
  no change  everything else: candidates marked "false" and all that was looked at
Also adds a "real_refined" layer to candidates.gpkg (the outlines used, with how
they were made) and writes label_preview.jpg.

    python benchmarks/make_labels.py --squares work_dirs/train_squares
    python benchmarks/make_labels.py --squares work_dirs/train_squares --no-sam
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from osgeo import gdal, ogr, osr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from score_testset import review_layers  # noqa: E402

gdal.UseExceptions()
ogr.UseExceptions()

PEFTCD = ROOT.parent / "PeftCD"
SAM_CKPT = PEFTCD / "sam2" / "sam2_hiera_large.pt"
CHANGE, IGNORE = 1, 255
DRAWN_SAM_MAX_M2 = 3000   # drawn buildings up to this size are tightened by SAM


def load_sam(device):
    sys.path.insert(0, str(PEFTCD))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(build_sam2("sam2_hiera_l.yaml", str(SAM_CKPT), device=device))


def read_rgb(path):
    ds = gdal.Open(str(path))
    return ds.ReadAsArray()[:3].transpose(1, 2, 0).copy(), ds


def sam_outline(pred, img, blob, ctx=96):
    """SAM's outline of the object under `blob` (bool mask), or None if it disagrees.

    SAM sees a full-resolution window around the blob, prompted with the blob's box
    and its most interior point.
    """
    ys, xs = np.nonzero(blob)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = blob.shape
    wy0, wy1 = max(0, y0 - ctx), min(h, y1 + ctx)
    wx0, wx1 = max(0, x0 - ctx), min(w, x1 + ctx)
    win = blob[wy0:wy1, wx0:wx1].astype(np.uint8)
    dist = cv2.distanceTransform(win, cv2.DIST_L2, 3)
    py, px = np.unravel_index(int(dist.argmax()), dist.shape)
    pred.set_image(img[wy0:wy1, wx0:wx1])
    masks, scores, _ = pred.predict(point_coords=np.array([[px, py]], np.float32),
                                    point_labels=np.array([1], np.int32),
                                    box=np.array([x0 - wx0, y0 - wy0, x1 - wx0, y1 - wy0], np.float32),
                                    multimask_output=True)
    best, best_iou = None, 0.0
    for m, s in zip(masks, scores):
        m = m.astype(bool)
        inter = (m & (win > 0)).sum()
        iou = inter / max((m | (win > 0)).sum(), 1)
        ratio = m.sum() / max(win.sum(), 1)
        if 0.33 <= ratio <= 3.0 and iou > best_iou:
            best, best_iou = m, iou
    if best is None or best_iou < 0.3:
        return None, best_iou
    best = tidy(best, win > 0)
    out = np.zeros_like(blob)
    out[wy0:wy1, wx0:wx1] = best
    return out, best_iou


def tidy(mask, blob, min_part=0.05):
    """Keep SAM's parts that touch the candidate, drop specks, fill holes."""
    m = mask.astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    biggest = st[1:, cv2.CC_STAT_AREA].max() if n > 1 else 0
    for i in range(1, n):
        part = cc == i
        if (part & blob).any() and st[i, cv2.CC_STAT_AREA] >= min_part * biggest:
            keep[i] = True
    m = keep[cc].astype(np.uint8)
    # Fill holes: background not connected to the window border.
    inv = (1 - m).astype(np.uint8)
    n, cc = cv2.connectedComponents(inv, connectivity=4)
    border = set(np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]])))
    holes = ~np.isin(cc, list(border)) & (inv > 0)
    return (m > 0) | holes


def polygons_to_layer(ds_out, name, mask_ids, info, like_ds):
    """Polygonise an id raster into layer `name` with attributes from info[id]."""
    h, w = mask_ids.shape
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Int32)
    mem.SetGeoTransform(like_ds.GetGeoTransform())
    mem.SetProjection(like_ds.GetProjection())
    mem.GetRasterBand(1).WriteArray(mask_ids.astype(np.int32))
    msk = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    msk.GetRasterBand(1).WriteArray((mask_ids > 0).astype(np.uint8))
    if ds_out.GetLayerByName(name) is not None:
        ds_out.DeleteLayer(name)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(like_ds.GetProjection())
    lyr = ds_out.CreateLayer(name, srs, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("oid", ogr.OFTInteger))
    for fld in ("type", "outline"):
        lyr.CreateField(ogr.FieldDefn(fld, ogr.OFTString))
    lyr.CreateField(ogr.FieldDefn("sam_iou", ogr.OFTReal))
    gdal.Polygonize(mem.GetRasterBand(1), msk.GetRasterBand(1), lyr, 0, ["8CONNECTED=8"])
    lyr.ResetReading()
    for f in lyr:
        t, how, iou = info[f.GetField("oid")]
        f.SetField("type", t)
        f.SetField("outline", how)
        f.SetField("sam_iou", round(float(iou), 2))
        lyr.SetFeature(f)


def make(square, pred, band_sam_m, band_rough_m, missed_r_m):
    a, ds = read_rgb(square / "before.tif")
    b, _ = read_rgb(square / "after.tif")
    gt = ds.GetGeoTransform()
    px = abs(gt[1])
    gpkg = square / "candidates.gpkg"
    rv = review_layers(ds, square)
    shape = rv["unsure"].shape
    label = np.zeros(shape, np.uint8)
    change = np.zeros(shape, bool)
    band = np.zeros(shape, bool)
    ids_out = np.zeros(shape, np.int32)
    info = {}
    n_sam = 0
    for ids, tps, origin in rv["real"]:
        for i, t in tps.items():
            blob = ids == i
            if not blob.any():
                continue
            out, iou, how = None, 0.0, "drawn" if origin == "drawn" else "candidate"
            # Your own drawings are kept as drawn, except single buildings (a rough shape
            # around one building is what SAM tightens well). Large areas, sites, car
            # parks and roads are outlined badly by SAM: it may take only part of them.
            use_sam = pred is not None
            if origin == "drawn":
                use_sam = use_sam and t == "building" and blob.sum() * px * px <= DRAWN_SAM_MAX_M2
            if use_sam:
                img = a if t == "demolition" else b
                out, iou = sam_outline(pred, img, blob)
                if out is not None:
                    how = "sam"
                    n_sam += 1
            region = out if out is not None else blob
            r = max(1, int(round((band_sam_m if out is not None else band_rough_m) / px)))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            ring = (cv2.dilate(region.astype(np.uint8), k) > 0) & ~(cv2.erode(region.astype(np.uint8), k) > 0)
            change |= region
            band |= ring
            oid = len(info) + 1
            ids_out[region & (ids_out == 0)] = oid
            info[oid] = (t, how, iou)
    label[change] = CHANGE
    label[rv["unsure"] | rv["unreviewed"] | rv["small"]] = IGNORE
    label[band] = IGNORE
    rr = max(1, int(round(missed_r_m / px)))
    for row, col, _ in rv["points"]:
        cv2.circle(label, (col, row), rr, IGNORE, -1)

    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(str(square / "train_label.tif"), label.shape[1], label.shape[0], 1, gdal.GDT_Byte,
                        options=["COMPRESS=DEFLATE"])
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(label)
    out_ds.GetRasterBand(1).SetNoDataValue(IGNORE)
    out_ds = None
    g = ogr.Open(str(gpkg), 1)
    polygons_to_layer(g, "real_refined", ids_out, info, ds)
    g = None

    prev = b.copy()
    prev[label == CHANGE] = (0.5 * prev[label == CHANGE] + [0, 110, 0]).astype(np.uint8)
    prev[label == IGNORE] = (0.5 * prev[label == IGNORE] + [60, 60, 60]).astype(np.uint8)
    s = 1600 / max(prev.shape[:2])
    cv2.imwrite(str(square / "label_preview.jpg"),
                cv2.cvtColor(cv2.resize(prev, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
    n = label.size
    return {"real": len(info), "sam": n_sam, "change_pct": 100 * (label == CHANGE).sum() / n,
            "ignore_pct": 100 * (label == IGNORE).sum() / n,
            "unreviewed_pct": 100 * rv["unreviewed"].sum() / n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squares", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--band-sam-m", type=float, default=1.0, help="ignore band around SAM outlines (m)")
    ap.add_argument("--band-rough-m", type=float, default=3.0, help="ignore band around candidate outlines (m)")
    ap.add_argument("--missed-r-m", type=float, default=10.0, help="ignore disc around missed points (m)")
    args = ap.parse_args()
    pred = None
    if not args.no_sam:
        import torch
        pred = load_sam("cuda" if torch.cuda.is_available() else "cpu")
    for sq in sorted(p for p in Path(args.squares).iterdir() if (p / "candidates.gpkg").is_file()):
        if args.only and sq.name not in args.only.split(","):
            continue
        s = make(sq, pred, args.band_sam_m, args.band_rough_m, args.missed_r_m)
        warn = (f"  ({s['unreviewed_pct']:.1f}% of the square under candidates not reviewed yet: ignored)"
                if s["unreviewed_pct"] >= 0.05 else "")
        print(f"{sq.name}: {s['real']} real changes ({s['sam']} outlined by SAM), "
              f"change {s['change_pct']:.1f}% / ignored {s['ignore_pct']:.1f}% of the square{warn}")


if __name__ == "__main__":
    main()
