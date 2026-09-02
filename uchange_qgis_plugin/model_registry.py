import os

_PLUGIN_DIR = os.path.realpath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_PLUGIN_DIR, ".."))

MODEL_REGISTRY = {
    "DINOv2 ViT-B (generalizable)":  {"file": "dinov2_vitb14_levir.pth", "type": "dinov2"},
    "DINOv2 ViT-B (fine-tuned)":     {"file": "dinov2_vitb14_egybcd.pth", "type": "dinov2"},
    "ChangerEx (R18) - LEVIR-CD (buildings)":   {"file": "ChangerEx_r18-512x512_40k_levircd.pth", "type": "opencd"},
    "SCD UPerNet (R18) - SECOND (land cover)":  {"file": "scd_upernet_r18_10k_second.pth", "type": "opencd_scd"},
}

DEFAULT_WEIGHTS = "dinov2_vitb14_egybcd.pth"

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


def resolve_weights_path(display_name):
    entry = MODEL_REGISTRY.get(display_name)
    if entry is None:
        raise ValueError(f"Unknown model: {display_name}")
    return os.path.join(_PROJECT_ROOT, entry["file"])
