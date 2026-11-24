"""OpenVINO for the DINOv2 passes on the CPU (optional: pip install openvino).

Against PyTorch's own CPU path (the decoder one tile at a time), the change model ran
1.46x faster and the false-alarm filter's backbone (280 px crops) 1.20x on a 6-core
laptop CPU, and 1.19x and 1.40x on a 24-thread CPU;
change probabilities stay within 1e-8 of PyTorch's (benchmarks/cpu_backends.py).
detect_changes uses it on the CPU when the package is installed; anything that fails falls
back to PyTorch.

Each model is converted once per model file and input size (about 20 s) and kept, about
0.35 GB, in openvino_cache/ beside the model file; later runs load it in about a second.
OpenVINO holds its own copy of the weights and needs more working memory than PyTorch's
tile-at-a-time decoder: a CPU run peaks about 0.7 GB higher (2.9 against 2.2 GB on a
500 m area). It gets the CPU batch of 4 tiles whole and the filter's crops 16 at a time,
the fastest on both machines (--groups: 4 and 8 tiles tie, 1 is slower; 16 and 32 crops
tie, 8 is a little slower).
"""
import contextlib
import hashlib
import os
import warnings

import torch

CACHE_VERSION = 1           # bump when the converted graphs change
TILES, CROPS = 4, 16        # inputs per OpenVINO call (see above)


def version():
    """The installed OpenVINO's version, or None if it is missing or does not load."""
    try:
        import openvino
        return openvino.__version__.split("-")[0]
    except Exception:       # a broken install can fail in other ways than ImportError
        return None


def _cache_file(weights_path, tag, cache_dir=None):
    import openvino
    st = os.stat(weights_path)
    key = "|".join(str(k) for k in (os.path.abspath(weights_path), st.st_size, st.st_mtime_ns, tag,
                                     torch.__version__, openvino.__version__, CACHE_VERSION))
    stem = os.path.splitext(os.path.basename(weights_path))[0]
    folder = cache_dir or os.path.join(os.path.dirname(os.path.abspath(weights_path)), "openvino_cache")
    return os.path.join(folder, f"{stem}.{tag}.{hashlib.sha1(key.encode()).hexdigest()[:10]}.xml")


def _save(model, xml):
    """Writes the converted model (.xml + .bin); replaces older conversions of the same
    model file and input (another torch or OpenVINO version, a changed file)."""
    import openvino as ov
    folder, name = os.path.split(xml)
    os.makedirs(folder, exist_ok=True)
    tmp = xml[:-4] + f".tmp{os.getpid()}.xml"
    ov.save_model(model, tmp, compress_to_fp16=False)
    os.replace(tmp[:-4] + ".bin", xml[:-4] + ".bin")
    os.replace(tmp, xml)                    # last: the .xml marks a complete pair
    prefix, keep = name.rsplit(".", 2)[0] + ".", name[:-4]
    for f in os.listdir(folder):
        if f.startswith(prefix) and f.rsplit(".", 1)[0] != keep and ".tmp" not in f:
            try:
                os.remove(os.path.join(folder, f))
            except OSError:
                pass


def _compile(make_module, example, shapes, weights_path, tag, threads, log, cache_dir=None):
    import openvino as ov
    core = ov.Core()
    xml = _cache_file(weights_path, tag, cache_dir)
    if os.path.isfile(xml):
        model = core.read_model(xml)
    else:
        log("  OpenVINO: converting the model for the CPU (once per model and tile size, ~20 s)")
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")     # tracer warnings about shapes fixed at export
            model = ov.convert_model(make_module(), example_input=example, input=shapes)
        try:
            _save(model, xml)
        except OSError as e:
            log(f"  OpenVINO: could not keep the converted model ({e}); it is converted again next run")
    # float32 throughout: on CPUs with bfloat16 units OpenVINO would otherwise pick bfloat16.
    return core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": int(threads),
                                             "INFERENCE_PRECISION_HINT": "f32",
                                             "PERFORMANCE_HINT": "LATENCY"})


class _Runner:
    def __init__(self, compiled):
        self.req, self.outputs = compiled.create_infer_request(), compiled.outputs

    def __call__(self, *xs, group=None):
        """Outputs for inputs xs (batch first), run `group` items at a time."""
        n = xs[0].shape[0]
        group = group or n
        parts = []
        for i in range(0, n, group):
            res = self.req.infer([x[i:i + group].detach().float().contiguous().numpy() for x in xs])
            # OpenVINO reuses its output buffers in the next call: copy.
            parts.append([torch.from_numpy(res[o].copy()) for o in self.outputs])
        return parts[0] if len(parts) == 1 else [torch.cat(p) for p in zip(*parts)]


@contextlib.contextmanager
def change_model(model, weights_path, tile_size, threads, log, cache_dir=None):
    """While open, the change model's forward runs in OpenVINO for tiles of tile_size px
    (other inputs go to PyTorch). Yields whether OpenVINO is in use."""
    try:
        import openvino as ov
        x = torch.zeros(2, 3, tile_size, tile_size)
        old, model.decoder_chunk = model.decoder_chunk, None    # one graph, any batch size
        try:
            compiled = _compile(lambda: model, (x, x), [ov.PartialShape([-1, 3, tile_size, tile_size])] * 2,
                                weights_path, f"change{tile_size}", threads, log, cache_dir)
        finally:
            model.decoder_chunk = old
        run = _Runner(compiled)
    except Exception as e:
        log(f"  OpenVINO did not load the change model ({e}); using PyTorch")
        yield False
        return
    torch_forward = model.forward

    def forward(img_a, img_b):
        if tuple(img_a.shape[1:]) != (3, tile_size, tile_size):
            return torch_forward(img_a, img_b)
        return run(img_a, img_b, group=TILES)[0]

    model.forward = forward
    try:
        yield True
    finally:
        del model.forward


class _Tokens(torch.nn.Module):
    def __init__(self, backbone, size, layers):
        super().__init__()
        self.bb, self.size, self.layers = backbone, int(size), tuple(int(l) for l in layers)

    def forward(self, x):
        from .falsealarm_filter import backbone_tokens
        return tuple(backbone_tokens(self.bb, x, self.size, self.layers))


def filter_tokens(backbone, weights_path, size, layers, threads, log, cache_dir=None):
    """The false-alarm filter's backbone pass (falsealarm_filter.backbone_tokens) in
    OpenVINO, for filter_mask(tokens_fn=...); None if OpenVINO does not load it."""
    size = int(size)
    try:
        import openvino as ov
        x = torch.zeros(2, 3, size, size)
        tag = f"filter{size}-" + "-".join(str(int(l)) for l in layers)
        compiled = _compile(lambda: _Tokens(backbone, size, layers).eval(), (x,),
                            [ov.PartialShape([-1, 3, size, size])], weights_path, tag, threads, log, cache_dir)
        run = _Runner(compiled)
    except Exception as e:
        log(f"  OpenVINO did not load the filter's backbone ({e}); using PyTorch")
        return None
    return lambda x: run(x, group=CROPS)
