import os
import sys
import glob
import ctypes

import numpy as np

from .model_registry import (
    MODEL_REGISTRY, DEFAULT_WEIGHTS,
    SECOND_SEMANTIC_CLASSES, SECOND_SEMANTIC_PALETTE,
    is_scd_model, resolve_weights_path,
)


def _ensure_venv_on_path():
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    try:
        from . import _env_config
        sp = _env_config.VENV_SITE_PACKAGES
        if sp and sp not in sys.path and pyver in sp:
            sys.path.insert(0, sp)
        return
    except ImportError:
        pass

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    venv_patterns = [
        os.path.join(project_root, "venv", "Lib", "site-packages"),
        os.path.join(project_root, "venv", "lib", pyver, "site-packages"),
    ]
    venv_paths = []
    for pat in venv_patterns:
        venv_paths.extend(glob.glob(pat))
    for sp in venv_paths:
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _preload_scipy_openblas():
    for sp in sys.path:
        libs_dir = os.path.join(sp, "scipy.libs")
        if not os.path.isdir(libs_dir):
            continue
        for so in sorted(glob.glob(os.path.join(libs_dir, "libscipy_openblas*.so"))):
            try:
                ctypes.cdll.LoadLibrary(so)
            except OSError:
                pass


def _preload_nvidia_libs():
    for sp in sys.path:
        nvidia_dir = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        lib_dirs = sorted(glob.glob(os.path.join(nvidia_dir, "*", "lib")))
        for lib_dir in lib_dirs:
            for so in sorted(glob.glob(os.path.join(lib_dir, "*.so*"))):
                try:
                    ctypes.cdll.LoadLibrary(so)
                except OSError:
                    pass
        break


_ensure_venv_on_path()
_preload_nvidia_libs()
_preload_scipy_openblas()

import torch

_model_cache = {}

_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(checkpoint_path, device=None, model_type="opencd"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_key = (os.path.abspath(checkpoint_path), str(device))
    if cache_key in _model_cache:
        return _model_cache[cache_key], "Model loaded from cache"

    from .models import build_changer_model, build_scd_model, build_dinov2_model

    if model_type == "opencd_scd":
        model, summary = build_scd_model(checkpoint_path, device)
    elif model_type == "dinov2":
        model, summary = build_dinov2_model(checkpoint_path, device)
    else:
        model, summary = build_changer_model(checkpoint_path, device)

    _model_cache[cache_key] = model
    return model, summary


_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _rgb_to_gray3ch(img):
    gray = np.dot(img, _GRAY_WEIGHTS)
    return np.stack([gray, gray, gray], axis=-1)


def normalize_tile(tile, grayscale=False):
    img = tile.astype(np.float32) / 255.0
    if grayscale:
        img = _rgb_to_gray3ch(img)
    return np.transpose((img - _NORM_MEAN) / _NORM_STD, (2, 0, 1))
