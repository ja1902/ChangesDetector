"""Is ONNX Runtime or OpenVINO faster than PyTorch for the change model on this CPU?

Times one batch of 4 tiles (252 px, the presets' size) the way the plugin runs it on the
CPU (PyTorch, the decoder one tile at a time), then the same model in ONNX Runtime and
OpenVINO if they are installed, whole and as backbone only (the backbone is also what the
false-alarm filter runs). Prints how far each one's change probabilities are from
PyTorch's.

    venv/bin/pip install onnxruntime onnx openvino "numpy<2"   # keep numpy 1: GDAL needs it
    venv/bin/python benchmarks/cpu_backends.py [--threads N] [--repeat 5] [--weights F.pth]
    venv/bin/python benchmarks/cpu_backends.py --groups     # OpenVINO: 1-8 tiles per call
    venv/bin/pip uninstall -y onnxruntime onnx openvino openvino-telemetry ml_dtypes \
        coloredlogs flatbuffers humanfriendly                    # afterwards: the plugin needs none

Without "numpy<2", installing onnx upgrades numpy to 2.x, and GDAL built against numpy 1
then fails ("No module named '_gdal_array'").

The first ONNX run exports the model (~10 s, ~0.4 GB in a temporary folder).
"""
import argparse
import atexit
import os
import platform
import shutil
import sys
import tempfile
import time
import types
import warnings

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Only the model code is needed (torch and numpy): load uchange_qgis_plugin/models without
# the plugin package's __init__, which needs GDAL.
_pkg = types.ModuleType("uchange_qgis_plugin")
_pkg.__path__ = [os.path.join(ROOT, "uchange_qgis_plugin")]
sys.modules.setdefault("uchange_qgis_plugin", _pkg)
from uchange_qgis_plugin.models import build_dinov2_model  # noqa: E402

TILE, BATCH = 252, 4


def cpu_name():
    """The CPU's name, and the vector instructions PyTorch uses on it (AVX2, AVX512...)."""
    name = platform.processor() or platform.machine()
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        else:
            with open("/proc/cpuinfo") as f:
                name = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration, ImportError):
        pass
    try:
        simd = torch.backends.cpu.get_cpu_capability()
    except AttributeError:                  # torch < 2.0
        simd = "unknown"
    return f"{name} (PyTorch uses: {simd})"


class Backbone(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        return tuple(self.m.extract_multilayer_features(x))


def export(module, args, path, names_in, names_out):
    kw = dict(input_names=names_in, output_names=names_out, opset_version=17,
              dynamic_axes={n: {0: "n"} for n in names_in + names_out})
    try:
        torch.onnx.export(module, args, path, dynamo=False, **kw)
    except TypeError:                       # torch < 2.5 has no dynamo argument
        torch.onnx.export(module, args, path, **kw)


def timed(fn, repeat):
    fn()                                    # warm-up
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


class Tokens(torch.nn.Module):
    """The false-alarm filter's backbone pass (falsealarm_filter.backbone_tokens)."""

    def __init__(self, model, size=280, layers=(8, 11)):
        super().__init__()
        self.bb, self.size, self.layers = model.backbone, size, layers

    def forward(self, x):
        bb, g = self.bb, self.size // 14
        x = bb.patch_embed(x)
        pe = bb._interpolate_pos_embed(x.shape[1])
        x = torch.cat([(bb.cls_token + pe[:, :1]).expand(x.shape[0], -1, -1), x + pe[:, 1:]], 1)
        out = []
        for i, blk in enumerate(bb.blocks):
            x = blk(x)
            if i in self.layers:
                out.append(x[:, 1:].reshape(x.shape[0], g, g, -1))
            if i >= max(self.layers):
                break
        return tuple(out)


def groups(model, nt, repeat):
    """OpenVINO (as the plugin converts it, straight from PyTorch) with more or fewer
    inputs per call: does more memory per call buy speed?"""
    import openvino as ov
    core = ov.Core()
    cfg = {"INFERENCE_NUM_THREADS": nt, "INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "LATENCY"}
    rng = np.random.default_rng(0)

    def convert(module, size, n_in):
        x = torch.zeros(2, 3, size, size)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            om = ov.convert_model(module, example_input=(x,) * n_in, input=[ov.PartialShape([-1, 3, size, size])] * n_in)
        return core.compile_model(om, "CPU", cfg).create_infer_request()

    n = 8
    a = rng.standard_normal((n, 3, TILE, TILE), dtype=np.float32)
    b = rng.standard_normal((n, 3, TILE, TILE), dtype=np.float32)
    ta, tb = torch.from_numpy(a), torch.from_numpy(b)
    print("converting the change model (~20 s)...", flush=True)
    model.decoder_chunk = None
    req = convert(model, TILE, 2)
    model.decoder_chunk = 1
    with torch.inference_mode():
        t_pt = timed(lambda: [model(ta[i:i + BATCH], tb[i:i + BATCH]) for i in range(0, n, BATCH)], repeat) / n
    print(f"  change model, PyTorch (the plugin's CPU path):   {t_pt:.3f} s per tile")
    for g in (1, 2, 4, 8):
        t = timed(lambda: [req.infer([a[i:i + g], b[i:i + g]]) for i in range(0, n, g)], repeat) / n
        print(f"  change model, OpenVINO, {g} tiles per call:       {t:.3f} s per tile   x{t_pt / t:.2f}", flush=True)

    n, size = 32, 280
    c = rng.standard_normal((n, 3, size, size), dtype=np.float32)
    tc = torch.from_numpy(c)
    print("converting the filter's backbone (~20 s)...", flush=True)
    tok = Tokens(model, size).eval()
    req = convert(tok, size, 1)
    with torch.inference_mode():
        t_pt = timed(lambda: tok(tc), repeat) / n
    print(f"  filter backbone, PyTorch (32 crops per call):    {t_pt:.3f} s per crop")
    for g in (4, 8, 16, 32):
        t = timed(lambda: [req.infer([c[i:i + g]]) for i in range(0, n, g)], repeat) / n
        print(f"  filter backbone, OpenVINO, {g:2d} crops per call:  {t:.3f} s per crop   x{t_pt / t:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=None, help="default: what the plugin uses")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--weights", default=os.path.join(ROOT, "dinov2_vitb14_finetuned.pth"))
    ap.add_argument("--groups", action="store_true",
                    help="OpenVINO only: time it with 1-8 tiles and 4-32 filter crops per call")
    args = ap.parse_args()

    # as detect_changes.cpu_threads: one fewer than the CPU has, if torch takes them all
    n, total = torch.get_num_threads(), os.cpu_count() or 1
    nt = args.threads or (max(1, total - 1) if n >= total and total > 2 else n)
    torch.set_num_threads(nt)
    print(f"CPU: {cpu_name()}, {os.cpu_count()} logical CPUs, {nt} threads; torch {torch.__version__}")

    model, _ = build_dinov2_model(args.weights, torch.device("cpu"))
    model.decoder_chunk = 1                 # as the plugin runs it on the CPU
    if args.groups:
        groups(model, nt, args.repeat)
        return
    rng = np.random.default_rng(0)
    a = rng.standard_normal((BATCH, 3, TILE, TILE), dtype=np.float32)
    b = rng.standard_normal((BATCH, 3, TILE, TILE), dtype=np.float32)
    ab = np.concatenate([a, b])
    ta, tb, tab = torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(ab)

    def prob(logits):
        return torch.softmax(torch.as_tensor(np.asarray(logits)).float(), 1)[:, 1].numpy()

    with torch.inference_mode():
        ref = prob(model(ta, tb))
        rows = [("PyTorch (what the plugin runs)", timed(lambda: model(ta, tb), args.repeat), 0.0)]
        t_bb = timed(lambda: model.extract_multilayer_features(tab), args.repeat)
    print(f"  {rows[0][0]:<44} {rows[0][1]:6.2f} s per {BATCH} tiles   (backbone alone {t_bb:.2f} s)")

    tmp = tempfile.mkdtemp(prefix="cd_onnx_")
    atexit.register(shutil.rmtree, tmp, True)
    full, bb = os.path.join(tmp, "model.onnx"), os.path.join(tmp, "backbone.onnx")
    try:
        import onnx  # noqa: F401
        model.decoder_chunk = None          # one graph for the whole batch
        warnings.filterwarnings("ignore")   # tracer warnings about shapes fixed at export
        export(model, (ta, tb), full, ["a", "b"], ["logits"])
        export(Backbone(model).eval(), (tab,), bb, ["x"], [f"f{i}" for i in range(4)])
        model.decoder_chunk = 1
    except ImportError:
        print("  (pip install onnx to export the model for ONNX Runtime / OpenVINO)")
        full = None

    def hybrid(feats_fn):
        """Backbone in the runtime, decoder in PyTorch one tile at a time."""
        def run():
            f = [torch.from_numpy(np.asarray(x)) for x in feats_fn()]
            with torch.inference_mode():
                return torch.cat([model.decoder([x[i:i + 1] for x in f], [x[BATCH + i:BATCH + i + 1] for x in f],
                                                out_size=(TILE, TILE)) for i in range(BATCH)])
        return run

    if full:
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads = nt, 1
            s_full = ort.InferenceSession(full, so, providers=["CPUExecutionProvider"])
            s_bb = ort.InferenceSession(bb, so, providers=["CPUExecutionProvider"])
            run_full = lambda: s_full.run(None, {"a": a, "b": b})[0]
            run_hyb = hybrid(lambda: s_bb.run(None, {"x": ab}))
            rows += [(f"ONNX Runtime {ort.__version__}, whole model", timed(run_full, args.repeat),
                      np.abs(prob(run_full()) - ref).max()),
                     ("ONNX Runtime backbone + PyTorch decoder", timed(run_hyb, args.repeat),
                      np.abs(prob(run_hyb()) - ref).max())]
            t = timed(lambda: s_bb.run(None, {"x": ab}), args.repeat)
            print(f"  {'ONNX Runtime backbone alone':<44} {t:6.2f} s")
        except ImportError:
            print("  (pip install onnxruntime to time ONNX Runtime)")
        try:
            import openvino as ov
            core = ov.Core()
            cfg = {"INFERENCE_NUM_THREADS": nt, "INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "LATENCY"}
            c_full, c_bb = core.compile_model(full, "CPU", cfg), core.compile_model(bb, "CPU", cfg)
            r_full, r_bb = c_full.create_infer_request(), c_bb.create_infer_request()
            run_full = lambda: r_full.infer({"a": a, "b": b})[c_full.output(0)].copy()
            run_hyb = hybrid(lambda: [r_bb.infer({"x": ab})[c_bb.output(i)].copy() for i in range(4)])
            rows += [(f"OpenVINO {ov.__version__.split('-')[0]}, whole model", timed(run_full, args.repeat),
                      np.abs(prob(run_full()) - ref).max()),
                     ("OpenVINO backbone + PyTorch decoder", timed(run_hyb, args.repeat),
                      np.abs(prob(run_hyb()) - ref).max())]
            t = timed(lambda: r_bb.infer({"x": ab}), args.repeat)
            print(f"  {'OpenVINO backbone alone':<44} {t:6.2f} s")
        except ImportError:
            print("  (pip install openvino to time OpenVINO)")

    print()
    base = rows[0][1]
    for name, t, diff in rows:
        print(f"{name:<46} {t:6.2f} s   x{base / t:4.2f}   max prob difference {diff:.1e}")


if __name__ == "__main__":
    main()
