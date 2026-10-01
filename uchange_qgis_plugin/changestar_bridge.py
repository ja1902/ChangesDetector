"""Optional Changen2 ChangeStar (ViT-L) model, used by the ensemble preset.

ChangeStar lives in the `torchange` package (Apache-2.0 code); its pretrained
weights (EVER-Z/Changen2-ChangeStar1x256 on Hugging Face, downloaded on first
use) are CC BY-NC-SA 4.0, i.e. non-commercial. The package is not a plugin
requirement; this module imports it only when the ensemble is selected.

Evaluated setting (research/RESULTS_2026-09.md): whole images up to 1024 px,
fp32, no flips; larger rasters run in 1024 px windows with 128 px overlap.
"""
import numpy as np

INSTALL_HINT = (
    "The 'Changen2 ViT-L + DINOv2' model needs the optional torchange package.\n"
    "Install it into the plugin's virtual environment:\n"
    "  pip install --no-deps torchange ever-beta\n"
    "  pip install einops timm albumentations tifffile tqdm wandb prettytable tensorboard matplotlib datasets huggingface_hub\n"
    "Its weights are licensed CC BY-NC-SA 4.0 (non-commercial use only)."
)


# import name -> pip package, for the "missing package" message
_PIP_NAMES = {"cv2": "opencv-python-headless", "skimage": "scikit-image", "PIL": "pillow",
              "sklearn": "scikit-learn", "yaml": "pyyaml",
              "segmentation_models_pytorch": "segmentation-models-pytorch"}


def import_error_message(exc):
    """Say what is actually missing: torchange itself, or one of its dependencies."""
    missing = (getattr(exc, "name", None) or "").split(".")[0]
    if missing == "torchange" or not missing:
        return INSTALL_HINT if missing else (
            "torchange is installed but could not be imported: %s: %s\n%s"
            % (type(exc).__name__, exc, INSTALL_HINT))
    return ("torchange is installed, but it needs '%s', which is missing.\n"
            "Install it into the plugin's virtual environment:\n  pip install %s"
            % (missing, _PIP_NAMES.get(missing, missing)))


def _ensure_strenum():
    """enum.StrEnum is new in Python 3.11; add a stand-in on 3.10."""
    import enum
    if not hasattr(enum, "StrEnum"):
        class StrEnum(str, enum.Enum):
            def __str__(self):
                return str(self.value)
        enum.StrEnum = StrEnum


def import_changen2():
    """Import torchange's Changen2 models, also on Python 3.10.

    torchange imports every one of its dataset modules on import, and one of
    them (BRIGHT, unused here) needs enum.StrEnum. Ubuntu 22.04 ships 3.10.
    """
    _ensure_strenum()
    import torchange.models.changen2 as c2
    return c2


HF_REPO = "EVER-Z/Changen2-ChangeStar1x256"
_WEIGHT_KEY = "s1_changestar_vitl_1x256"

OFFLINE_HINT = (
    "The Changen2 weights (%s on Hugging Face, ~1.2 GB) are not in this computer's cache "
    "and could not be downloaded.\n"
    "Either connect to the internet and run the model once, or run install.sh on a connected "
    "machine (it downloads them) and copy that machine's ~/.cache/huggingface folder here.\n"
    "Without them, use the 'DINOv2 ViT-B + synthetic data' model, which needs no download."
    % HF_REPO)


def prefetch_weights():
    """Download the Changen2 weights into the Hugging Face cache (needs internet).

    torchange fetches them on first use; running this at install time means the
    model also works later on a computer without internet access.
    """
    import json
    from huggingface_hub import hf_hub_download
    cfg = json.load(open(hf_hub_download(HF_REPO, "config.json"), encoding="utf-8"))
    return hf_hub_download(HF_REPO, cfg[_WEIGHT_KEY])


def build_changestar(device):
    try:
        c2 = import_changen2()
    except ImportError as exc:
        raise RuntimeError(import_error_message(exc)) from exc
    try:
        model = c2.s1_init_s1c1_changestar_vitl_1x256().to(device).eval()
    except Exception as exc:
        # no internet and nothing cached: huggingface_hub raises its own error types
        if type(exc).__module__.startswith("huggingface_hub") or isinstance(exc, (OSError, ConnectionError)):
            raise RuntimeError("%s: %s\n%s" % (type(exc).__name__, exc, OFFLINE_HINT)) from exc
        raise
    for p in model.parameters():
        p.requires_grad = False
    return model, "Model: Changen2 ChangeStar ViT-L (EVER-Z/Changen2-ChangeStar1x256, CC BY-NC-SA 4.0)"


def _windows(size, window, overlap):
    if size <= window:
        return [0]
    step = window - overlap
    starts = list(range(0, size - window + 1, step))
    if starts[-1] + window < size:
        starts.append(size - window)
    return starts


def changestar_probs(model, pre_img, post_img, device, window=1024, overlap=128,
                     progress_fn=None, cancel_fn=None):
    """uint8 HWC pair -> float32 change-probability map at the same size."""
    import torch
    import torch.nn.functional as F
    from .model_bridge import normalize_tile

    h, w = pre_img.shape[:2]
    ys, xs = _windows(h, window, overlap), _windows(w, window, overlap)
    coords = [(y, x) for y in ys for x in xs]
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    for i, (y, x) in enumerate(coords):
        if cancel_fn and cancel_fn():
            return None
        a = pre_img[y:y + window, x:x + window]
        b = post_img[y:y + window, x:x + window]
        th, tw = a.shape[:2]
        t = torch.from_numpy(np.concatenate([normalize_tile(a), normalize_tile(b)], 0))[None]
        t = t.float().to(device)
        ph, pw = (-th) % 32, (-tw) % 32
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode="reflect")
        with torch.inference_mode():
            out = model(t)["change_prediction"][0, 0, :th, :tw].float().cpu().numpy()
        acc[y:y + th, x:x + tw] += out
        cnt[y:y + th, x:x + tw] += 1.0
        if progress_fn:
            progress_fn(i + 1, len(coords))
    return acc / np.maximum(cnt, 1.0)


if __name__ == "__main__":
    _ensure_strenum()
    print("Downloaded:", prefetch_weights())
