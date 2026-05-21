import os
import sys
import glob

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


_ensure_venv_on_path()

import torch

_PLUGIN_DIR = os.path.realpath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_PLUGIN_DIR, ".."))
_CHANGEMAMBA_ROOT = os.path.join(_PROJECT_ROOT, "ChangeMamba")
def _find_peftcd_root():
    for dirpath, _, filenames in os.walk(os.path.join(_PROJECT_ROOT, "PeftCD-master")):
        if "DINO3CD.py" in filenames:
            return dirpath
    return os.path.join(_PROJECT_ROOT, "PeftCD-master")

_PEFTCD_ROOT = _find_peftcd_root()

MODEL_REGISTRY = {
    "MambaBCD - LEVIR-CD+ (buildings)":       {"file": "MambaBCD_Small_LEVIRCD+.pth",       "type": "mamba"},
    "MambaBCD - SYSU (vegetation / general)":  {"file": "MambaBCD_Small_SYSU.pth", "type": "mamba"},
    "PeftCD - LEVIR-CD+ (buildings)":          {"file": "PeftCD_LEVIRCD.ckpt",               "type": "peftcd"},
    "PeftCD - SYSU (vegetation / general)":    {"file": "PeftCD_SYSU.ckpt",                  "type": "peftcd"},
}

DEFAULT_WEIGHTS = "MambaBCD_Small_LEVIRCD+.pth"

_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_VSSM_COMMON = dict(
    patch_size=4, in_chans=3, num_classes=1000,
    ssm_d_state=1, ssm_ratio=2.0, ssm_rank_ratio=2.0,
    ssm_dt_rank="auto", ssm_act_layer="silu",
    ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
    ssm_init="v0", forward_type="v3noz",
    mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0,
    patch_norm=True, norm_layer="ln",
    downsample_version="v3", patchembed_version="v2",
    gmlp=False, use_checkpoint=False,
)

VSSM_CONFIGS = {
    "tiny":  {**_VSSM_COMMON, "depths": [2, 2, 4, 2],  "dims": 96,  "drop_path_rate": 0.2},
    "small": {**_VSSM_COMMON, "depths": [2, 2, 15, 2], "dims": 96,  "drop_path_rate": 0.3},
    "base":  {**_VSSM_COMMON, "depths": [2, 2, 15, 2], "dims": 128, "drop_path_rate": 0.6},
}

_model_cache = {}


class PeftCDWrapper(torch.nn.Module):
    """Wraps PeftCD model to average the two output logit tensors into one."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pre, post):
        outs = self.model(pre, post)
        return (outs[0] + outs[1]) / 2.0


def resolve_weights_path(display_name):
    entry = MODEL_REGISTRY.get(display_name)
    if entry is None:
        raise ValueError(f"Unknown model: {display_name}")
    return os.path.join(_PROJECT_ROOT, entry["file"])


def _model_type_for_path(checkpoint_path):
    """Determine model type from checkpoint extension."""
    if checkpoint_path.endswith(".ckpt"):
        return "peftcd"
    return "mamba"


def _ensure_changemamba_on_path():
    if _CHANGEMAMBA_ROOT not in sys.path:
        sys.path.insert(0, _CHANGEMAMBA_ROOT)


def _detect_model_size(checkpoint_path):
    name = os.path.basename(checkpoint_path).lower()
    for size in ("tiny", "small", "base"):
        if size in name:
            return size
    return "small"


def _build_mamba_model(checkpoint_path, device):
    _ensure_changemamba_on_path()
    from changedetection.models.ChangeMambaBCD import ChangeMambaBCD
    from changedetection.checkpoints import load_model_weights

    size = _detect_model_size(checkpoint_path)
    kwargs = VSSM_CONFIGS[size]

    model = ChangeMambaBCD(pretrained=None, **kwargs)
    load_info = load_model_weights(model, checkpoint_path)

    total_keys = len(model.state_dict())
    loaded = load_info["loaded_keys"]
    missing = len(load_info["missing_keys"])
    unexpected = len(load_info["unexpected_keys"])
    mismatched = len(load_info["mismatched_keys"])

    summary = (
        f"Model: MambaBCD-{size.title()} | "
        f"Checkpoint: {loaded}/{total_keys} keys loaded, "
        f"{missing} missing, {unexpected} unexpected, {mismatched} mismatched"
    )
    if loaded == 0:
        raise RuntimeError(
            f"No weights loaded — architecture mismatch. "
            f"First 5 unexpected: {load_info['unexpected_keys'][:5]}"
        )

    model.to(device).eval()
    return model, summary


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


def _build_peftcd_model(checkpoint_path, device):
    if _PEFTCD_ROOT not in sys.path:
        sys.path.insert(0, _PEFTCD_ROOT)

    _import_with_standard_hook('accelerate', 'transformers', 'peft')
    _patch_mmseg_version_check()

    prev_cwd = os.getcwd()
    os.chdir(_PEFTCD_ROOT)
    try:
        from DINO3CD import DINO3CD
        model = DINO3CD(peft_method='lora')
    finally:
        os.chdir(prev_cwd)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)

    cleaned = {}
    for k, v in state_dict.items():
        key = k[len('change_detection.'):] if k.startswith('change_detection.') else k
        cleaned[key] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()

    wrapped = PeftCDWrapper(model)
    wrapped.eval()

    summary = (
        f"Model: PeftCD (DINOv3+LoRA) | "
        f"Checkpoint: {os.path.basename(checkpoint_path)} | "
        f"Keys: {len(cleaned)}"
    )
    return wrapped, summary


def build_model(checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_key = (os.path.abspath(checkpoint_path), str(device))
    if cache_key in _model_cache:
        return _model_cache[cache_key], "Model loaded from cache"

    model_type = _model_type_for_path(checkpoint_path)
    if model_type == "peftcd":
        model, summary = _build_peftcd_model(checkpoint_path, device)
    else:
        model, summary = _build_mamba_model(checkpoint_path, device)

    _model_cache[cache_key] = model
    return model, summary


def normalize_tile(tile):
    img = tile.astype(np.float32) / 255.0
    return np.transpose((img - _NORM_MEAN) / _NORM_STD, (2, 0, 1))
