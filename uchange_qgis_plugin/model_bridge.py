import os
import sys
import glob
import ctypes

import numpy as np


def _ensure_venv_on_path():
    try:
        from . import _env_config
        sp = _env_config.VENV_SITE_PACKAGES
        if sp and sp not in sys.path:
            sys.path.insert(0, sp)
        return
    except ImportError:
        pass

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    venv_patterns = [
        os.path.join(project_root, "venv", "Lib", "site-packages"),
        os.path.join(project_root, "venv", "lib", "python*", "site-packages"),
    ]
    venv_paths = []
    for pat in venv_patterns:
        venv_paths.extend(glob.glob(pat))
    for sp in venv_paths:
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _preload_scipy_openblas():
    """Preload scipy's bundled OpenBLAS .so before scipy is imported.

    scipy wheels bundle their own OpenBLAS in a ``scipy.libs/`` directory
    next to the ``scipy/`` package.  When running inside QGIS the dynamic
    linker cannot find it because LD_LIBRARY_PATH does not include that
    directory.  Force-loading the library with ctypes makes the symbols
    available for all subsequent imports."""
    for sp in sys.path:
        libs_dir = os.path.join(sp, "scipy.libs")
        if not os.path.isdir(libs_dir):
            continue
        for so in sorted(glob.glob(os.path.join(libs_dir, "libscipy_openblas*.so"))):
            try:
                ctypes.cdll.LoadLibrary(so)
            except OSError:
                pass


_ensure_venv_on_path()
_preload_scipy_openblas()

import torch

_PLUGIN_DIR = os.path.realpath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_PLUGIN_DIR, ".."))
_OPENCD_ROOT = os.path.join(_PROJECT_ROOT, "open-cd-main")

MODEL_REGISTRY = {
    "ChangerEx (R18) - LEVIR-CD (buildings)":  {"file": "ChangerEx_r18-512x512_40k_levircd.pth", "type": "opencd"},
    "SCD UPerNet (R18) - SECOND (land cover)": {"file": "scd_upernet_r18_10k_second.pth", "type": "opencd_scd"},
}

DEFAULT_WEIGHTS = "ChangerEx_r18-512x512_40k_levircd.pth"

SECOND_SEMANTIC_CLASSES = (
    'unchanged', 'water', 'ground',
    'low vegetation', 'tree', 'building',
    'sports field',
)
SECOND_SEMANTIC_PALETTE = (
    (255, 255, 255), (0, 0, 255), (128, 128, 128),
    (0, 128, 0), (0, 255, 0), (128, 0, 0),
    (255, 0, 0),
)


def is_scd_model(display_name):
    entry = MODEL_REGISTRY.get(display_name)
    return entry is not None and entry.get("type") == "opencd_scd"

_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model_cache = {}


class OpenCDWrapper(torch.nn.Module):
    """Wraps an Open-CD model to accept (pre, post) ImageNet-normalized RGB
    tensors and return 2-class logits (B, 2, H, W)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pre, post):
        B, C, H, W = pre.shape
        inputs = torch.cat([pre, post], dim=1)
        batch_img_metas = [{'ori_shape': (H, W), 'img_shape': (H, W),
                           'pad_shape': (H, W), 'padding_size': [0, 0, 0, 0]}] * B
        return self.model.encode_decode(inputs, batch_img_metas)


class OpenCDSCDWrapper(torch.nn.Module):
    """Wraps an Open-CD SCD model to accept (pre, post) ImageNet-normalized RGB
    tensors and return a dict with binary logits + semantic logits."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_scd = True

    def forward(self, pre, post):
        B, C, H, W = pre.shape
        inputs = torch.cat([pre, post], dim=1)
        batch_img_metas = [{'ori_shape': (H, W), 'img_shape': (H, W),
                           'pad_shape': (H, W), 'padding_size': [0, 0, 0, 0]}] * B
        return self.model.encode_decode(inputs, batch_img_metas)


def resolve_weights_path(display_name):
    entry = MODEL_REGISTRY.get(display_name)
    if entry is None:
        raise ValueError(f"Unknown model: {display_name}")
    return os.path.join(_PROJECT_ROOT, entry["file"])


def _import_with_standard_hook(*module_names):
    """Import modules using Python's standard import, bypassing QGIS's custom
    import hook which breaks circular imports in accelerate/transformers."""
    import builtins
    import importlib
    qgis_import = builtins.__import__
    try:
        try:
            from qgis.utils import _builtin_import
            builtins.__import__ = _builtin_import
        except ImportError:
            pass
        for name in module_names:
            importlib.import_module(name)
    finally:
        builtins.__import__ = qgis_import


def _patch_mmseg_version_check():
    """mmseg checks mmcv < 2.2.0 at import time, but mmcv 2.2.0 is the only
    version with prebuilt wheels for torch 2.4. Temporarily spoof the version
    so the assertion passes, then restore it."""
    import mmcv
    real = mmcv.__version__
    mmcv.__version__ = '2.1.0'
    _import_with_standard_hook('mmseg')
    mmcv.__version__ = real


_OPENCD_CONFIG = "configs/changer/changer_ex_r18_512x512_40k_levircd.py"
_OPENCD_SCD_CONFIG = "configs/general_scd/scd_upernet_r18_256x512_10k_second.py"


def _ensure_scipy_optimize():
    """Make sure ``from scipy.optimize import linear_sum_assignment`` won't
    crash when mmseg.models is first imported.

    mmseg's HungarianAssigner does a top-level ``from scipy.optimize import
    linear_sum_assignment``.  If the venv's scipy is too new for the
    available numpy (e.g. scipy >= 1.14 with numpy < 2.0), that import
    crashes.  ChangerEx never uses HungarianAssigner, so a minimal stub
    is safe."""
    if 'scipy.optimize' in sys.modules:
        return
    try:
        import scipy.optimize  # noqa: F401
    except (ImportError, AttributeError):
        import types
        sys.modules.setdefault('scipy', types.ModuleType('scipy'))
        opt = types.ModuleType('scipy.optimize')
        opt.linear_sum_assignment = None
        sys.modules['scipy.optimize'] = opt


def _build_opencd_model(checkpoint_path, device):
    if _OPENCD_ROOT not in sys.path:
        sys.path.insert(0, _OPENCD_ROOT)

    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import DefaultScope

    _ensure_scipy_optimize()
    _patch_mmseg_version_check()
    _import_with_standard_hook(
        'opencd.models.backbones.interaction_resnet',
        'opencd.models.change_detectors.dual_input_encoder_decoder',
        'opencd.models.decode_heads.changer',
        'opencd.models.data_preprocessor',
    )

    cfg = Config.fromfile(os.path.join(_OPENCD_ROOT, _OPENCD_CONFIG))
    if hasattr(cfg.model, "pretrained"):
        cfg.model.pretrained = None
    for attr in ("backbone", "image_encoder", "decode_head", "auxiliary_head"):
        sub = getattr(cfg.model, attr, None)
        if sub is not None and hasattr(sub, "init_cfg"):
            sub.init_cfg = None

    from opencd.registry import MODELS
    DefaultScope.get_instance('opencd', scope_name='opencd')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location="cpu", logger=None)
    model = revert_sync_batchnorm(model)
    model.to(device).eval()

    wrapped = OpenCDWrapper(model)
    wrapped.eval()

    summary = (
        f"Model: ChangerEx (R18) | "
        f"Checkpoint: {os.path.basename(checkpoint_path)}"
    )
    return wrapped, summary


def _build_opencd_scd_model(checkpoint_path, device):
    if _OPENCD_ROOT not in sys.path:
        sys.path.insert(0, _OPENCD_ROOT)

    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import DefaultScope

    _ensure_scipy_optimize()
    _patch_mmseg_version_check()
    _import_with_standard_hook(
        'opencd.models.backbones.interaction_resnet',
        'opencd.models.change_detectors.siamencoder_decoder',
        'opencd.models.change_detectors.siamencoder_multidecoder',
        'opencd.models.decode_heads.general_scd_head',
        'opencd.models.decode_heads.multi_head',
        'opencd.models.necks.feature_fusion',
        'opencd.models.data_preprocessor',
    )

    cfg = Config.fromfile(os.path.join(_OPENCD_ROOT, _OPENCD_SCD_CONFIG))
    if hasattr(cfg.model, "pretrained"):
        cfg.model.pretrained = None
    for attr in ("backbone", "decode_head", "auxiliary_head"):
        sub = getattr(cfg.model, attr, None)
        if sub is not None and hasattr(sub, "init_cfg"):
            sub.init_cfg = None

    from opencd.registry import MODELS
    DefaultScope.get_instance('opencd', scope_name='opencd')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location="cpu", logger=None)
    model = revert_sync_batchnorm(model)
    model.to(device).eval()

    wrapped = OpenCDSCDWrapper(model)
    wrapped.eval()

    summary = (
        f"Model: SCD UPerNet (R18) | "
        f"Checkpoint: {os.path.basename(checkpoint_path)}"
    )
    return wrapped, summary


def build_model(checkpoint_path, device=None, model_type="opencd"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_key = (os.path.abspath(checkpoint_path), str(device))
    if cache_key in _model_cache:
        return _model_cache[cache_key], "Model loaded from cache"

    if model_type == "opencd_scd":
        model, summary = _build_opencd_scd_model(checkpoint_path, device)
    else:
        model, summary = _build_opencd_model(checkpoint_path, device)

    _model_cache[cache_key] = model
    return model, summary


def normalize_tile(tile):
    img = tile.astype(np.float32) / 255.0
    return np.transpose((img - _NORM_MEAN) / _NORM_STD, (2, 0, 1))
