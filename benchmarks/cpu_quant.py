r"""Would bfloat16 or INT8 make the OpenVINO CPU path faster, and how far do they move results?

Converts the change model (252 px tiles) and the false-alarm filter's backbone (280 px
crops) the way uchange_qgis_plugin/openvino_cpu.py does, then times and compares:

  f32        what the plugin runs now
  bf16       OpenVINO's bfloat16 (only on CPUs with bfloat16 units: AVX512_BF16 or AMX)
  int8       NNCF post-training quantization (transformer mode: LayerNorm, softmax and
             GELU stay in float), calibrated on real tiles; each block's second MLP
             layer (mlp.fc2) stays in float too: its input carries DINOv2's outlier
             activations, and quantizing it wrecks the results (all change pixels lost,
             filter tokens at cosine 0.34 to f32)
  int8+bf16  the INT8 model with its float parts in bfloat16 (bfloat16 CPUs only)

Tiles come from a real before/after pair (any two aligned images of the same area, about
0.5 m per pixel): random tiles calibrate INT8; other random tiles, the ones where f32
finds the most change, measure the differences against f32. Reported per variant: speed against f32,
largest and mean change in probability, and how many pixels cross 0.6 (the blob
threshold) either way; for the filter, the cosine similarity of the tokens to f32's.
Single pass, no flips: the flips use the same model.

    venv/bin/pip install openvino nncf "numpy<2"   # keep numpy 1: GDAL needs it
    venv/bin/python benchmarks/cpu_quant.py [--pair before.jpg after.jpg] [--threads N]
    venv/bin/python benchmarks/cpu_quant.py --save work_dirs/exp_quant   # keep the INT8 models
    venv\Scripts\python benchmarks\cpu_quant.py --pair before.tif after.tif    # Windows

Takes about 5-15 minutes (the two INT8 calibrations are most of it).

Speed means little on a CPU without VNNI (AVX512_VNNI or AVX_VNNI): INT8 is emulated
there. The device's capabilities are printed first.
"""
import argparse
import os
import sys
import time
import types
import warnings

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Only the model code is needed: skip the plugin package's __init__, which needs GDAL.
_pkg = types.ModuleType("uchange_qgis_plugin")
_pkg.__path__ = [os.path.join(ROOT, "uchange_qgis_plugin")]
sys.modules.setdefault("uchange_qgis_plugin", _pkg)
from uchange_qgis_plugin.models import build_dinov2_model  # noqa: E402
from uchange_qgis_plugin.openvino_cpu import CROPS, TILES, _Tokens  # noqa: E402

TILE, CROP, LAYERS, CUT = 252, 280, (8, 11), 0.6
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_pair(before, after):
    Image.MAX_IMAGE_PIXELS = None
    a = np.asarray(Image.open(before).convert("RGB"))
    b = np.asarray(Image.open(after).convert("RGB"))
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def norm(t):
    return np.transpose((t.astype(np.float32) / 255 - MEAN) / STD, (2, 0, 1))


def windows(a, b, size, n, rng):
    """n random aligned windows (n, 3, size, size) from both dates."""
    h, w = a.shape[:2]
    ys, xs = rng.integers(0, h - size, n), rng.integers(0, w - size, n)
    pa = np.stack([norm(a[y:y + size, x:x + size]) for y, x in zip(ys, xs)])
    pb = np.stack([norm(b[y:y + size, x:x + size]) for y, x in zip(ys, xs)])
    return pa, pb


def timed(fn, repeat):
    fn()
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def run(req, inputs, group):
    n, out = inputs[0].shape[0], []
    for i in range(0, n, group):
        res = req.infer([x[i:i + group] for x in inputs])
        out.append([res[o].copy() for o in req.model_outputs])
    return [np.concatenate(p) for p in zip(*out)]


def prob(logits):
    l = logits.astype(np.float64)
    return 1 / (1 + np.exp(l[:, 0] - l[:, 1]))


_OUT = []                   # --out file: gets the results, not NNCF's progress bars


def say(*parts, **_):
    print(*parts, flush=True)
    for f in _OUT:
        print(*parts, file=f, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, default=[os.path.join(ROOT, "pair1", "before.jpg"),
                                                os.path.join(ROOT, "pair1", "after.jpg")])
    ap.add_argument("--weights", default=os.path.join(ROOT, "dinov2_vitb14_finetuned.pth"))
    ap.add_argument("--threads", type=int, default=None, help="default: what the plugin uses")
    ap.add_argument("--calib", type=int, default=64, help="calibration tiles / crops")
    ap.add_argument("--test", type=int, default=24, help="tiles / crops compared and timed")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--save", default=None, help="folder to keep the INT8 models in")
    ap.add_argument("--out", default=None, help="also write the results to this file")
    args = ap.parse_args()
    if args.out:
        _OUT.append(open(args.out, "w", encoding="utf-8"))
    for f in args.pair:
        if not os.path.isfile(f):
            sys.exit(f"{f} not found: pass a before/after pair with --pair BEFORE AFTER")

    import logging
    import nncf
    import openvino as ov
    nncf.set_log_level(logging.ERROR)

    n, total = torch.get_num_threads(), os.cpu_count() or 1
    nt = args.threads or (max(1, total - 1) if n >= total and total > 2 else n)
    torch.set_num_threads(nt)
    core = ov.Core()
    caps = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
    say(f"CPU: {core.get_property('CPU', 'FULL_DEVICE_NAME')}, {total} logical CPUs, {nt} threads")
    say(f"OpenVINO {ov.__version__.split('-')[0]}, NNCF {nncf.__version__}; capabilities: {' '.join(caps)}")
    has_bf16 = "BF16" in caps
    if not has_bf16:
        say("  no bfloat16 units: bf16 variants skipped")

    model, _ = build_dinov2_model(args.weights, torch.device("cpu"))
    model.decoder_chunk = None
    a_img, b_img = load_pair(*args.pair)
    rng = np.random.default_rng(0)

    def convert(module, example, shapes):
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ov.convert_model(module, example_input=example, input=shapes)

    def compile_(om, precision):
        return core.compile_model(om, "CPU", {"INFERENCE_NUM_THREADS": nt, "INFERENCE_PRECISION_HINT": precision,
                                              "PERFORMANCE_HINT": "LATENCY"}).create_infer_request()

    def quantize(om, items):
        return nncf.quantize(om, nncf.Dataset(items), model_type=nncf.ModelType.TRANSFORMER,
                             preset=nncf.QuantizationPreset.MIXED, subset_size=len(items),
                             ignored_scope=nncf.IgnoredScope(patterns=[r".*mlp\.fc2.*"]))

    def variants(om, items, name):
        say(f"quantizing the {name} on {len(items)} samples...", flush=True)
        t = time.perf_counter()
        q = quantize(om, items)
        say(f"  {time.perf_counter() - t:.0f} s", flush=True)
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            ov.save_model(q, os.path.join(args.save, f"{name.replace(' ', '_')}_int8.xml"), compress_to_fp16=False)
        v = [("f32", om, "f32"), ("int8", q, "f32")]
        if has_bf16:
            v[1:1] = [("bf16", om, "bf16")]
            v.append(("int8+bf16", q, "bf16"))
        return [(label, compile_(m, p)) for label, m, p in v]

    # change model
    x = torch.zeros(2, 3, TILE, TILE)
    om = convert(model, (x, x), [ov.PartialShape([-1, 3, TILE, TILE])] * 2)
    ca, cb = windows(a_img, b_img, TILE, args.calib, rng)
    ta, tb = windows(a_img, b_img, TILE, 4 * args.test, rng)
    reqs = variants(om, [(ca[i:i + 1], cb[i:i + 1]) for i in range(len(ca))], "change model")
    keep = np.argsort(-(prob(run(reqs[0][1], (ta, tb), TILES)[0]) >= CUT).mean((1, 2)), kind="stable")[:args.test]
    ta, tb = ta[keep], tb[keep]
    say(f"\nchange model, {args.test} tiles of {TILE} px, {TILES} per call")
    ref, t_ref = None, None
    for label, req in reqs:
        t = timed(lambda: run(req, (ta, tb), TILES), args.repeat) / args.test
        p = prob(run(req, (ta, tb), TILES)[0])
        if ref is None:
            ref, t_ref = p, t
            say(f"  {label:<10} {t:.3f} s per tile             {(ref >= CUT).mean() * 100:.2f}% of pixels >= {CUT}")
            continue
        d = np.abs(p - ref)
        up, down = ((p >= CUT) & (ref < CUT)).sum(), ((p < CUT) & (ref >= CUT)).sum()
        say(f"  {label:<10} {t:.3f} s per tile  x{t_ref / t:.2f}   prob diff max {d.max():.3f} "
              f"mean {d.mean():.4f} p99.9 {np.quantile(d, 0.999):.3f}; crossing {CUT}: +{up} -{down} "
              f"of {(ref >= CUT).sum()} px", flush=True)

    # filter backbone
    x = torch.zeros(2, 3, CROP, CROP)
    om = convert(_Tokens(model.backbone, CROP, LAYERS).eval(), (x,), [ov.PartialShape([-1, 3, CROP, CROP])])
    cc = np.concatenate(windows(a_img, b_img, CROP, args.calib // 2, rng))
    tc = np.concatenate(windows(a_img, b_img, CROP, args.test // 2, rng))
    reqs = variants(om, [(cc[i:i + 1],) for i in range(len(cc))], "filter backbone")
    say(f"\nfilter backbone, {len(tc)} crops of {CROP} px, {CROPS} per call")
    ref, t_ref = None, None
    for label, req in reqs:
        t = timed(lambda: run(req, (tc,), CROPS), args.repeat) / len(tc)
        toks = run(req, (tc,), CROPS)
        if ref is None:
            ref, t_ref = toks, t
            say(f"  {label:<10} {t:.3f} s per crop")
            continue
        cos = []
        for r, q in zip(ref, toks):
            r, q = r.reshape(-1, r.shape[-1]), q.reshape(-1, q.shape[-1])
            cos.append((r * q).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(q, axis=1) + 1e-12))
        sims = " ".join(f"block {l}: mean {c.mean():.4f} min {c.min():.3f}" for l, c in zip(LAYERS, cos))
        say(f"  {label:<10} {t:.3f} s per crop  x{t_ref / t:.2f}   token cosine to f32, {sims}", flush=True)


if __name__ == "__main__":
    main()
