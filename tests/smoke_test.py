#!/usr/bin/env python3
"""Smoke test: the pipeline end to end on a small synthetic pair, plus the pieces that broke before.

Needs the installed model weights (run the installer first). No test framework:

    venv/bin/python tests/smoke_test.py            # all checks (GPU if available, ~1-2 min)
    venv/bin/python tests/smoke_test.py --quick    # skip the runs of detect_changes.py

Prints PASS / FAIL per check; the exit code is the number of failures.
"""
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
gdal.UseExceptions()

GSD, SIZE, X0, Y0 = 0.5, 400, 500000.0, 4000000.0     # 200 m square, UTM 31N
RESULTS = []


def check(name):
    def wrap(fn):
        def run(*a):
            try:
                fn(*a)
                RESULTS.append((name, None))
                print(f"PASS  {name}", flush=True)
            except Exception as e:                        # report and go on
                RESULTS.append((name, e))
                print(f"FAIL  {name}: {e}", flush=True)
                traceback.print_exc(limit=2)
        return run
    return wrap


def write_tif(path, rgb, georef=True):
    ds = gdal.GetDriverByName("GTiff").Create(str(path), rgb.shape[1], rgb.shape[0], 3, gdal.GDT_Byte)
    if georef:
        ds.SetGeoTransform((X0, GSD, 0, Y0 + SIZE * GSD, 0, -GSD))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32631)
        ds.SetProjection(srs.ExportToWkt())
    for k in range(3):
        ds.GetRasterBand(k + 1).WriteArray(rgb[..., k])
    ds = None


def synthetic_pair(folder):
    """Textured ground with houses; the after image adds three new buildings and a cleared plot."""
    rng = np.random.default_rng(0)
    import cv2
    ground = cv2.GaussianBlur(rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32), (0, 0), 6)
    base = np.clip(120 + 400 * ground[..., None] * np.array([1.0, 0.9, 0.7]), 0, 255).astype(np.uint8)
    base = (base + rng.integers(0, 12, base.shape)).clip(0, 255).astype(np.uint8)
    for y, x in rng.integers(20, SIZE - 60, (14, 2)):                     # existing houses
        cv2.rectangle(base, (int(x), int(y)), (int(x) + 24, int(y) + 18), (200, 90, 70), -1)
    after = base.copy()
    for (y, x), c in zip(((60, 300), (250, 80), (320, 300)), ((235, 235, 230), (210, 205, 200), (180, 60, 50))):
        cv2.rectangle(after, (x, y), (x + 40, y + 30), c, -1)             # new buildings
    cv2.rectangle(after, (150, 150), (230, 210), (170, 150, 110), -1)    # cleared plot
    write_tif(folder / "before.tif", base)
    write_tif(folder / "after.tif", after)


def run_cli(folder, *extra):
    out = folder / f"out_{abs(hash(extra)) % 10**6}"
    cmd = [sys.executable, str(ROOT / "detect_changes.py"), "--before", str(folder / "before.tif"),
           "--after", str(folder / "after.tif"), "--output", str(out), "--output-gpkg", str(out) + ".gpkg",
           *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return p, Path(str(out) + ".gpkg")


def features(gpkg):
    ds = ogr.Open(str(gpkg))
    lay = ds.GetLayer(0)
    names = [lay.GetLayerDefn().GetFieldDefn(i).GetName() for i in range(lay.GetLayerDefn().GetFieldCount())]
    rows = [({n: f.GetField(n) for n in names}, f.GetGeometryRef().Clone()) for f in lay]
    gtype = lay.GetGeomType()
    return names, rows, gtype


# ---------------------------------------------------------------- pieces, no model needed

@check("grouping: pieces of one filtered change share an id")
def t_grouping():
    import cv2
    from uchange_qgis_plugin.raster_io import _group_by_filter
    m = np.zeros((20, 40), np.uint8)
    lab = np.zeros((20, 40), np.int32)
    m[2:5, 2:5] = 1; m[2:5, 8:11] = 1; lab[1:6, 1:12] = 3          # two pieces of change 3
    m[10:13, 2:5] = 1; m[10:13, 8:11] = 1                          # two pieces with no filter change
    m[15:19, 20:30] = 1; lab[15:19, 20:23] = 5; lab[15:19, 23:30] = 7
    m[2:5, 30:33] = 1; lab[2:5, 30:33] = 7
    n, ids = cv2.connectedComponents(m, connectivity=8)
    new, n2 = _group_by_filter(ids, n, lab)
    assert n2 - 1 == 4, f"expected 4 changes, got {n2 - 1}"
    assert new[3, 3] == new[3, 9] and new[11, 3] != new[11, 9] and new[16, 25] == new[3, 31]
    assert sorted(np.unique(new)) == list(range(n2))


@check("polygons: one MultiPolygon feature per change, area kept")
def t_polygons(tmp):
    from uchange_qgis_plugin.raster_io import polygonize_mask
    m = np.zeros((100, 100), np.uint8)
    m[10:30, 10:30] = 1; m[10:30, 40:60] = 1                     # two pieces of one change
    m[70:90, 70:90] = 1                                          # another change
    lab = np.zeros((100, 100), np.int32)
    lab[5:35, 5:65] = 1; lab[65:95, 65:95] = 2
    gt = (X0, GSD, 0, Y0 + 50, 0, -GSD)
    srs = osr.SpatialReference(); srs.ImportFromEPSG(32631)
    total, final = polygonize_mask(m, gt, srs.ExportToWkt(), str(tmp / "p.gpkg"),
                                   filter_labels=lab, filter_real=np.array([np.nan, 0.8, 0.6]))
    names, rows, gtype = features(tmp / "p.gpkg")
    assert (total, final, len(rows)) == (2, 2, 2), (total, final, len(rows))
    assert gtype == ogr.wkbMultiPolygon
    assert abs(sum(r[1].GetArea() for r in rows) - 3 * 400 * GSD * GSD) < 1.0
    assert {round(r[0]["filter_score"], 2) for r in rows} == {0.8, 0.6}


@check("area: crop_to_extent cuts both rasters, refuses a miss and an ungeoreferenced raster")
def t_crop(tmp, pair):
    from uchange_qgis_plugin.raster_io import crop_to_extent
    b, a, used, _ = crop_to_extent(str(pair / "before.tif"), str(pair / "after.tif"),
                                   (X0 + 50, Y0 + 50, X0 + 150, Y0 + 150), str(tmp))
    for p in (b, a):
        ds = gdal.Open(p)
        assert (ds.RasterXSize, ds.RasterYSize) == (200, 200), (ds.RasterXSize, ds.RasterYSize)
    for ext in ((0, 0, 10, 10),):
        try:
            crop_to_extent(str(pair / "before.tif"), str(pair / "after.tif"), ext, str(tmp))
            raise AssertionError("an area outside the raster was accepted")
        except ValueError:
            pass
    write_tif(tmp / "nogeo.tif", np.zeros((64, 64, 3), np.uint8), georef=False)
    try:
        crop_to_extent(str(tmp / "nogeo.tif"), str(tmp / "nogeo.tif"), (0, -10, 10, 0), str(tmp))
        raise AssertionError("an ungeoreferenced raster was accepted")
    except ValueError:
        pass


@check("filter: batched DINOv2 passes give the same features as one at a time")
def t_batch(pair):
    import torch
    import detect_changes as dc
    from uchange_qgis_plugin.falsealarm_filter import _Features
    from uchange_qgis_plugin.model_bridge import build_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, _ = build_model(str(ROOT / "dinov2_vitb14_finetuned.pth"), dev, model_type="dinov2")
    A, _ = dc.read_image(str(pair / "before.tif"))
    B, _ = dc.read_image(str(pair / "after.tif"))
    pix = [tuple(np.mgrid[y:y + 20, x:x + 30].reshape(2, -1)) for y, x in ((40, 50), (200, 210), (300, 90))]
    fx = _Features(m.backbone, dev, 280, (8, 11))
    together = fx.blobs(A, B, pix, GSD)
    alone = [fx.blob(A, B, ys, xs, GSD) for ys, xs in pix]
    assert all(np.allclose(t, a, atol=1e-4) for t, a in zip(together, alone))


def t_openvino(tmp, pair):
    from uchange_qgis_plugin import openvino_cpu
    if openvino_cpu.version() is None:
        print("SKIP  openvino: not installed")
        return
    _t_openvino(tmp, pair)


@check("openvino: the CPU backend gives PyTorch's change probabilities and filter features")
def _t_openvino(tmp, pair):
    import torch
    import detect_changes as dc
    from uchange_qgis_plugin import openvino_cpu
    from uchange_qgis_plugin.falsealarm_filter import _Features
    from uchange_qgis_plugin.model_bridge import build_model
    cpu, weights = torch.device("cpu"), str(ROOT / "dinov2_vitb14_finetuned.pth")
    m, _ = build_model(weights, cpu, model_type="dinov2")
    m.decoder_chunk = 1
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((3, 2, 3, 252, 252), dtype=np.float32))
    with torch.inference_mode():
        ref = torch.softmax(m(x[:, 0], x[:, 1]), 1)[:, 1]
        with openvino_cpu.change_model(m, weights, 252, torch.get_num_threads(), print) as used:
            assert used, "OpenVINO did not load the model"
            out = torch.softmax(m(x[:, 0], x[:, 1]), 1)[:, 1]
    assert "forward" not in m.__dict__ and not m.training, "the model was not restored"
    assert (out - ref).abs().max() < 1e-5, f"probabilities differ by {(out - ref).abs().max():.1e}"
    A, _ = dc.read_image(str(pair / "before.tif"))
    B, _ = dc.read_image(str(pair / "after.tif"))
    pix = [tuple(np.mgrid[y:y + 20, x:x + 30].reshape(2, -1)) for y, x in ((40, 50), (200, 210))]
    fn = openvino_cpu.filter_tokens(m.backbone, weights, 280, (8, 11), torch.get_num_threads(), print)
    assert fn is not None, "OpenVINO did not load the filter's backbone"
    ov = _Features(m.backbone, cpu, 280, (8, 11), fn).blobs(A, B, pix, GSD)
    pt = _Features(m.backbone, cpu, 280, (8, 11)).blobs(A, B, pix, GSD)
    if openvino_cpu.has_bf16():         # the filter's backbone runs in bfloat16 there
        cos = min(float(((o * p).sum(-1) / (np.linalg.norm(o, axis=-1) * np.linalg.norm(p, axis=-1))).min())
                  for o, p in zip(ov, pt))
        assert cos > 0.999, f"bfloat16 filter features: cosine to float32 down to {cos:.4f}"
    else:
        diff = max(float(np.abs(o - p).max()) for o, p in zip(ov, pt))
        assert diff < 1e-3, f"filter features differ by {diff:.1e}"


# ---------------------------------------------------------------- detect_changes.py end to end

@check("CLI: default run writes one feature per change with its scores")
def t_cli_default(pair):
    p, gpkg = run_cli(pair)
    assert p.returncode == 0, p.stdout[-500:] + p.stderr[-500:]
    names, rows, gtype = features(gpkg)
    for f in ("id", "change", "confidence", "filter_score", "area"):
        assert f in names, f"missing field {f}"
    assert gtype == ogr.wkbMultiPolygon
    assert rows, "no change found on the synthetic pair (the checks below would pass vacuously)"
    ids = [r[0]["id"] for r in rows]
    assert len(ids) == len(set(ids)), "an id appears on more than one feature"
    assert "set by the false-alarm filter" in p.stdout, "threshold line should defer to the filter"


@check("CLI: --extent processes only the area, and rejects an area outside the image")
def t_cli_extent(pair):
    p, gpkg = run_cli(pair, "--extent", f"{X0 + 100},{Y0 + 100},{X0 + 200},{Y0 + 200}")
    assert p.returncode == 0, p.stdout[-500:] + p.stderr[-500:]
    assert "Image size: 200x200" in p.stdout, "the area was not cut out"
    box = ogr.CreateGeometryFromWkt(f"POLYGON(({X0 + 99} {Y0 + 99},{X0 + 201} {Y0 + 99},{X0 + 201} {Y0 + 201},"
                                    f"{X0 + 99} {Y0 + 201},{X0 + 99} {Y0 + 99}))")
    assert all(box.Contains(g) for _, g in features(gpkg)[1]), "a change lies outside the area"
    p, _ = run_cli(pair, "--extent", "0,0,10,10")
    assert p.returncode != 0 and "Area to process" in p.stdout


@check("CLI: --fast runs the false-alarm filter on 2 rotations")
def t_cli_fast(pair):
    p, gpkg = run_cli(pair, "--fast")
    assert p.returncode == 0, p.stdout[-500:] + p.stderr[-500:]
    assert "need both rotations" in p.stdout, "the filter did not run on 2 rotations"
    assert features(gpkg)[1], "no change found in fast mode"


@check("CLI: --min-area-m2 drops smaller changes")
def t_cli_min_area(pair):
    p, gpkg = run_cli(pair, "--min-area-m2", "1000000")
    assert p.returncode == 0, p.stdout[-500:] + p.stderr[-500:]
    assert "Minimum change area: 1e+06 m2" in p.stdout
    assert len(features(gpkg)[1]) == 0


def main():
    quick = "--quick" in sys.argv
    if not (ROOT / "dinov2_vitb14_finetuned.pth").is_file():
        print("Model weights not found: run the installer first.")
        return 1
    with tempfile.TemporaryDirectory(prefix="smoke_") as d:
        tmp = Path(d)
        pair = tmp / "pair"
        pair.mkdir()
        synthetic_pair(pair)
        t_grouping()
        t_polygons(tmp)
        t_crop(tmp, pair)
        t_batch(pair)
        t_openvino(tmp, pair)
        if not quick:
            t_cli_default(pair)
            t_cli_extent(pair)
            t_cli_min_area(pair)
            t_cli_fast(pair)
    failed = [n for n, e in RESULTS if e is not None]
    print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed"
          + (": " + "; ".join(failed) if failed else ""))
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
