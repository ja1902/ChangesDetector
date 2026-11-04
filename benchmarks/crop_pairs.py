#!/usr/bin/env python3
"""Align the two dates of each prepared scene and cut random squares to review.

Whole Google Earth captures are several km2 each: too much to check by hand. This
takes the scenes written by prep_pairs.py, removes the shift between the two dates
(Google's dates are often a few metres apart) and cuts --per-pair randomly placed
squares of --km side. Random, so the test is not tilted towards places where change
is easy to see. The same seed gives the same squares.

    python benchmarks/prep_pairs.py --src captures/ --out scenes/ --assume-gsd 0.38
    python benchmarks/crop_pairs.py --scenes scenes/ --out testset/ --km 1 --per-pair 1

Writes testset/<scene>_c<i>/before.tif, after.tif and testset/crops.csv.
"""
import argparse
import csv
import zlib
from pathlib import Path

import cv2
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

LOCAL_MARGIN = 48       # px the shift of a square may differ from the scene's


def read_rgb(path):
    ds = gdal.Open(str(path))
    return ds.ReadAsArray()[:3].transpose(1, 2, 0), ds


def gray(img):
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)


def date_shift(a, b):
    """(dx, dy) to add to a's pixel coordinates to find the same ground in b."""
    f = 4
    ga = cv2.resize(gray(a), None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
    gb = cv2.resize(gray(b), None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
    win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb, win)
    dx, dy = dx * f, dy * f
    # Refine at full resolution on a central window, after the coarse shift.
    h, w = a.shape[:2]
    s = min(2048, h // 2, w // 2)
    y0, x0 = (h - s) // 2, (w - s) // 2
    ix, iy = int(round(dx)), int(round(dy))
    pa = gray(a[y0:y0 + s, x0:x0 + s])
    pb = gray(b[y0 + iy:y0 + iy + s, x0 + ix:x0 + ix + s])
    if pb.shape == pa.shape:
        (rx, ry), resp = cv2.phaseCorrelate(pa, pb, cv2.createHanningWindow((s, s), cv2.CV_32F))
        if abs(rx) < 8 and abs(ry) < 8:
            dx, dy = ix + rx, iy + ry
    return int(round(dx)), int(round(dy))


def write(path, img, gt, proj):
    h, w = img.shape[:2]
    ds = gdal.GetDriverByName("GTiff").Create(str(path), w, h, 3, gdal.GDT_Byte,
                                              options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    for i in range(3):
        ds.GetRasterBand(i + 1).WriteArray(img[..., i])
    ds = None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", required=True, help="output folder of prep_pairs.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--km", type=float, default=1.0, help="side of each square (km)")
    ap.add_argument("--per-pair", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in sorted(p for p in Path(args.scenes).iterdir() if (p / "before.tif").is_file()):
        a, ds = read_rgb(scene / "before.tif")
        b, _ = read_rgb(scene / "after.tif")
        gt, proj = ds.GetGeoTransform(), ds.GetProjection()
        gsd = abs(gt[1])
        dx, dy = date_shift(a, b)
        h, w = a.shape[:2]
        # Region of a whose shifted counterpart lies inside b, with room for the
        # local shift to differ from the scene's (patchwork dates, hills, lean).
        m = LOCAL_MARGIN
        vx0, vy0 = max(0, -dx) + m, max(0, -dy) + m
        vx1, vy1 = min(w, w - dx) - m, min(h, h - dy) - m
        side = int(round(args.km * 1000 / gsd))
        if side > min(vx1 - vx0, vy1 - vy0):
            side = min(vx1 - vx0, vy1 - vy0)
            print(f"{scene.name}: scene smaller than {args.km} km, using {side * gsd / 1000:.2f} km squares")
        rng = np.random.default_rng(args.seed + zlib.crc32(scene.name.encode()))
        boxes = []
        for _ in range(1000):
            if len(boxes) == args.per_pair:
                break
            x = int(rng.integers(vx0, vx1 - side + 1))
            y = int(rng.integers(vy0, vy1 - side + 1))
            if all(x + side <= bx or bx + side <= x or y + side <= by or by + side <= y for bx, by in boxes):
                boxes.append((x, y))
        if len(boxes) < args.per_pair:
            print(f"{scene.name}: only room for {len(boxes)} non-overlapping squares")
        for i, (x, y) in enumerate(boxes, start=1):
            name = f"{scene.name}_c{i}"
            d = out / name
            d.mkdir(exist_ok=True)
            cgt = (gt[0] + x * gt[1], gt[1], 0.0, gt[3] + y * gt[5], 0.0, gt[5])
            ca = a[y:y + side, x:x + side]
            # Local shift: the scene's, corrected on this square.
            lx, ly = dx, dy
            for _ in range(2):
                ex, ey = date_shift(ca, b[y + ly:y + ly + side, x + lx:x + lx + side])
                if abs(ex) > m or abs(ey) > m:
                    break
                lx, ly = int(np.clip(lx + ex, dx - m, dx + m)), int(np.clip(ly + ey, dy - m, dy + m))
            cb = b[y + ly:y + ly + side, x + lx:x + lx + side]
            rx, ry = date_shift(ca, cb)
            write(d / "before.tif", ca, cgt, proj)
            write(d / "after.tif", cb, cgt, proj)
            rows.append({"crop": name, "scene": scene.name, "x_px": x, "y_px": y, "side_px": side,
                         "gsd_m": f"{gsd:.2f}", "area_km2": f"{(side * gsd) ** 2 / 1e6:.3f}",
                         "scene_shift_px": f"{dx},{dy}", "square_shift_px": f"{lx},{ly}",
                         "residual_shift_px": f"{rx},{ry}"})
            print(f"{name}: square at ({x}, {y}), {side * gsd / 1000:.2f} km; dates shifted by ({lx}, {ly}) px "
                  f"here (scene: {dx}, {dy}); left after alignment: ({rx}, {ry})")
    with open(out / "crops.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n{len(rows)} squares in {out}")


if __name__ == "__main__":
    main()
