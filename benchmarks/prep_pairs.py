#!/usr/bin/env python3
"""Turn Google Earth Pro captures into georeferenced test pairs.

Each pair is a folder holding the two captures of one site, taken from the same,
unmoved view (so both share one pixel grid), plus the georeferencing of either
image in one of these forms (first found wins):

  *.points          the GCP file QGIS's Georeferencer saves ("Save GCP points");
                    any transform type, only the points are used
  world file        .jgw / .pgw / .wld / .tfw next to one of the images (+ .prj or --crs)
  bounds.txt        "N S E W" in degrees, from Google Earth's Add > Image Overlay dialog
  gsd.txt           just the pixel size in metres (e.g. measured on a tennis court, 23.77 m
                    long), or --assume-gsd for every pair without georeferencing: enough for
                    testing (sizes and areas are right), but the pair sits at a made-up
                    place on the map, so it will not line up with basemaps

The same georeferencing is applied to both images, which are then warped onto one
grid in their local UTM zone (true metres; Web Mercator overstates the pixel size
by 1/cos(latitude)). Dates are read from the file names (2018, 2018-05, 2018_05_21);
the earlier capture is "before".

    python benchmarks/prep_pairs.py --src captures/ --out testset/

Writes testset/<pair>/before.tif, after.tif and testset/pairs.csv (one row per pair;
columns you fill in by hand are kept when the script is re-run).
"""
import argparse
import csv
import math
import re
import sys
import zlib
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
WORLD_EXT = {".jgw", ".jpgw", ".pgw", ".pngw", ".wld", ".tfw", ".tifw"}
AUTO_COLS = ["pair", "before_file", "after_file", "before_date", "after_date", "width_px", "height_px",
             "gsd_m", "area_km2", "utm_epsg", "georef", "gcp_rms_px"]
HAND_COLS = ["before_provider", "after_provider", "before_season", "after_season",
             "lean_before", "lean_after", "notes"]
DATE_RE = re.compile(r"((?:19|20)\d\d)(?:[-_.]?(0[1-9]|1[0-2]))?(?:[-_.]?(0[1-9]|[12]\d|3[01]))?")


def date_key(name):
    m = DATE_RE.search(name)
    if not m:
        return None
    y, mo, d = m.group(1), m.group(2) or "", m.group(3) or ""
    return "-".join(x for x in (y, mo, d) if x)


def srs_from(text, default):
    srs = osr.SpatialReference()
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if text:
        srs.SetFromUserInput(text)
    else:
        srs.SetFromUserInput(default)
    return srs


def georef_from_points(path, default_crs):
    """Affine geotransform fitted to a QGIS .points file; returns (gt, srs, rms in map units)."""
    crs_text, rows = None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#CRS:"):
            crs_text = line[5:].strip() or None
            continue
        if line.startswith("#") or line.lower().startswith("mapx"):
            continue
        f = line.split(",")
        if len(f) >= 4 and (len(f) < 5 or f[4].strip() != "0"):
            rows.append([float(v) for v in f[:4]])
    if len(rows) < 3:
        sys.exit(f"{path}: need at least 3 enabled points, found {len(rows)}")
    r = np.array(rows)
    mx, my, sx, sy = r.T
    px, py = sx, -sy                       # QGIS stores the pixel row as a negative y
    a = np.stack([np.ones_like(px), px, py], 1)
    cx, *_ = np.linalg.lstsq(a, mx, rcond=None)
    cy, *_ = np.linalg.lstsq(a, my, rcond=None)
    res = np.hypot(a @ cx - mx, a @ cy - my)
    gt = (cx[0], cx[1], cx[2], cy[0], cy[1], cy[2])
    return gt, srs_from(crs_text, default_crs), float(np.sqrt((res ** 2).mean())), len(rows)


def georef_from_world(path, default_crs):
    v = [float(x) for x in Path(path).read_text().split()[:6]]
    a, d, b, e, c, f = v                   # world file: centre of the top-left pixel
    gt = (c - a / 2 - b / 2, a, b, f - d / 2 - e / 2, d, e)
    prj = None
    for cand in (path.with_suffix(".prj"), path.parent / (path.stem + ".prj")):
        if cand.is_file():
            prj = cand.read_text()
    return gt, srs_from(prj, default_crs)


def georef_from_bounds(path, w, h):
    n, s, e, wst = [float(x) for x in Path(path).read_text().replace(",", " ").split()[:4]]
    gt = (wst, (e - wst) / w, 0.0, n, 0.0, -(n - s) / h)
    return gt, srs_from("EPSG:4326", "EPSG:4326")


def utm_for(gt, srs, w, h):
    """EPSG code of the UTM zone at the image centre."""
    x = gt[0] + gt[1] * w / 2 + gt[2] * h / 2
    y = gt[3] + gt[4] * w / 2 + gt[5] * h / 2
    wgs = srs_from("EPSG:4326", "EPSG:4326")
    lon, lat, _ = osr.CoordinateTransformation(srs, wgs).TransformPoint(x, y)
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def find_georef(folder, images, default_crs, w, h, assume_gsd=None):
    pts = sorted(folder.glob("*.points"))
    if pts:
        gt, srs, rms, n = georef_from_points(pts[0], default_crs)
        return gt, srs, f"gcp ({n} points)", rms
    for img in images:
        for ext in WORLD_EXT:
            for cand in (img.with_suffix(ext), folder / (img.name + ext.lstrip("."))):
                if cand.is_file():
                    gt, srs = georef_from_world(cand, default_crs)
                    return gt, srs, "world file", None
    if (folder / "bounds.txt").is_file():
        gt, srs = georef_from_bounds(folder / "bounds.txt", w, h)
        return gt, srs, "bounds", None
    gsd = None
    if (folder / "gsd.txt").is_file():
        gsd = float((folder / "gsd.txt").read_text().split()[0])
    elif assume_gsd:
        gsd = assume_gsd
    if gsd:
        # Made-up place: UTM 31N, one pair per 10 km so pairs never overlap.
        x0 = 400000 + 10000 * (zlib.crc32(folder.name.encode()) % 50)
        gt = (float(x0), gsd, 0.0, 4500000.0, 0.0, -gsd)
        return gt, srs_from("EPSG:32631", "EPSG:32631"), "pixel size only", None
    return None


def warp_pair(src_imgs, out_paths, gt, srs, utm, gsd):
    """Same georeferencing and the same output grid for both images."""
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(utm)
    bounds = None
    for src, out in zip(src_imgs, out_paths):
        mem = gdal.Translate("", str(src), format="MEM", bandList=[1, 2, 3] if
                             gdal.Open(str(src)).RasterCount >= 3 else None)
        mem.SetGeoTransform(gt)
        mem.SetProjection(srs.ExportToWkt())
        opts = dict(format="GTiff", dstSRS=dst.ExportToWkt(), xRes=gsd, yRes=gsd,
                    resampleAlg="cubic", creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])
        if bounds is None:
            ds = gdal.Warp(str(out), mem, targetAlignedPixels=True, **opts)
            g = ds.GetGeoTransform()
            bounds = (g[0], g[3] + g[5] * ds.RasterYSize, g[0] + g[1] * ds.RasterXSize, g[3])
            size = (ds.RasterXSize, ds.RasterYSize)
        else:
            ds = gdal.Warp(str(out), mem, outputBounds=bounds, **opts)
        ds = None
    return size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with one sub-folder per pair")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crs", default="EPSG:3857",
                    help="CRS of the georeferencing when the file does not say (default: Web "
                         "Mercator, the CRS of the Google / Esri basemaps in QGIS)")
    ap.add_argument("--assume-gsd", type=float, default=None,
                    help="pixel size (m) for pairs with no georeferencing and no gsd.txt")
    ap.add_argument("--gsd", type=float, default=None,
                    help="output pixel size in m (default: the capture's own, rounded to 1 cm)")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "pairs.csv"
    old = {}
    if csv_path.is_file():
        with open(csv_path, newline="") as f:
            old = {r["pair"]: r for r in csv.DictReader(f)}

    rows, problems = [], []
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT
                      and not p.name.endswith("_modified.tif"))
        if len(imgs) != 2:
            problems.append(f"{folder.name}: expected 2 images, found {len(imgs)} ({', '.join(i.name for i in imgs)})")
            continue
        dates = [date_key(i.stem) for i in imgs]
        if all(dates) and dates[0] != dates[1]:
            imgs, dates = zip(*sorted(zip(imgs, dates), key=lambda t: t[1]))
        else:
            problems.append(f"{folder.name}: no dates in the file names, taking {imgs[0].name} as before")
            dates = dates if any(dates) else ("", "")
        d0, d1 = gdal.Open(str(imgs[0])), gdal.Open(str(imgs[1]))
        w, h = d0.RasterXSize, d0.RasterYSize
        if (d1.RasterXSize, d1.RasterYSize) != (w, h):
            problems.append(f"{folder.name}: the two images differ in size ({w}x{h} vs "
                            f"{d1.RasterXSize}x{d1.RasterYSize}): not the same view, skipped")
            continue
        g = find_georef(folder, imgs, args.crs, w, h, args.assume_gsd)
        if g is None:
            problems.append(f"{folder.name}: no .points, world file, bounds.txt or gsd.txt, skipped")
            continue
        gt, srs, how, rms = g
        utm = utm_for(gt, srs, w, h)
        # Native pixel size in metres, measured in UTM.
        tr = osr.CoordinateTransformation(srs, srs_from(f"EPSG:{utm}", "EPSG:4326"))
        corners = [(gt[0] + gt[1] * x + gt[2] * y, gt[3] + gt[4] * x + gt[5] * y) for x, y in ((0, 0), (w, 0), (0, h))]
        (x0, y0, _), (x1, y1, _), (x2, y2, _) = [tr.TransformPoint(*c) for c in corners]
        gsd_native = 0.5 * (math.hypot(x1 - x0, y1 - y0) / w + math.hypot(x2 - x0, y2 - y0) / h)
        gsd = args.gsd or round(gsd_native, 2)
        dst = out / folder.name
        dst.mkdir(exist_ok=True)
        if how == "pixel size only":
            # Nothing to reproject: keep the pixels exactly as captured.
            gsd = gt[1]
            for img, name in zip(imgs, ("before.tif", "after.tif")):
                gdal.Translate(str(dst / name), str(img), format="GTiff", outputSRS="EPSG:32631",
                               outputBounds=[gt[0], gt[3], gt[0] + gsd * w, gt[3] - gsd * h],
                               bandList=[1, 2, 3] if gdal.Open(str(img)).RasterCount >= 3 else None,
                               creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])
            ow, oh = w, h
        else:
            ow, oh = warp_pair(imgs, [dst / "before.tif", dst / "after.tif"], gt, srs, utm, gsd)
        rms_px = None
        if rms is not None:
            rms_px = rms / (0.5 * (math.hypot(gt[1], gt[4]) + math.hypot(gt[2], gt[5])))
            if rms_px > 3:
                problems.append(f"{folder.name}: GCP fit error {rms_px:.1f} px; check the points")
        row = {"pair": folder.name, "before_file": imgs[0].name, "after_file": imgs[1].name,
               "before_date": dates[0], "after_date": dates[1], "width_px": ow, "height_px": oh,
               "gsd_m": f"{gsd:.2f}", "area_km2": f"{ow * oh * gsd * gsd / 1e6:.3f}", "utm_epsg": utm,
               "georef": how, "gcp_rms_px": "" if rms_px is None else f"{rms_px:.2f}"}
        for c in HAND_COLS:
            row[c] = old.get(folder.name, {}).get(c, "")
        rows.append(row)
        print(f"{folder.name}: {dates[0] or '?'} -> {dates[1] or '?'}  {ow}x{oh} px  {gsd:.2f} m  "
              f"{float(row['area_km2']):.2f} km2  [{how}]")

    for name, r in old.items():                # keep rows of pairs not in this run
        if name not in {x["pair"] for x in rows}:
            rows.append(r)
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=AUTO_COLS + HAND_COLS)
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: r["pair"]))
    print(f"\n{len(rows)} pairs in {csv_path}")
    for p in problems:
        print("  ! " + p)


if __name__ == "__main__":
    main()
