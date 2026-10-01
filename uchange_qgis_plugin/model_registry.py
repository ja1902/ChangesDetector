import os

_PLUGIN_DIR = os.path.realpath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_PLUGIN_DIR, ".."))

# Inference presets: the exact pipeline each model was evaluated with in
# research/RESULTS_2026-09.md, so its recommended threshold means the same
# thing here. Thresholds are on logit-adjusted probabilities and were chosen on
# validation data from unfamiliar imagery (EGY-BCD / S2Looking val), not LEVIR.
#   scale         upsampling applied before tiling (DINOv2 was run at 1.4x)
#   tta           average the four flips
#   logit_adjust  remove the LEVIR change prior (pi = 0.0584) from the logits
#   target_gsd    resample coarser imagery up to this pixel size (metres)
PRESETS = {
    "synthetic": {
        "tile_size": 252, "overlap": 64, "scale": 1.4, "tta": True,
        "logit_adjust": True, "target_gsd": 0.5, "threshold": 0.70,
        # speckle removal (cleanup.py): off until its settings are validated
        "core_threshold": 0.0, "min_blob_m2": 0.0, "min_width_m": 0.0,
        # Fast mode (no flip averaging) offered in the dialog. No separate
        # "fast_threshold" yet: 0.70 is reused until one is measured
        # (scripts/e2026/e10_fast_mode.py).
        "fast_mode": True,
    },
    "ensemble": {
        "tile_size": 252, "overlap": 64, "scale": 1.4, "tta": True,
        "logit_adjust": True, "target_gsd": 0.5, "threshold": 0.35,
        "core_threshold": 0.0, "min_blob_m2": 0.0, "min_width_m": 0.0,
    },
}

MODEL_REGISTRY = {
    # Semantic mode. Change from the recommended binary model, before/after
    # classes from a land-cover head on the same DINOv2 backbone.
    "DINOv2 ViT-B + land cover, from -> to (recommended)": {
        "file": "dinov2_vitb14_c2s1_levir.pth", "type": "dinov2_lc", "preset": "synthetic",
        "landcover_head": "landcover_dinov2_vitb14_oem_second.pth",
        "tile_size": 252, "overlap": 64},
    "DINOv2 ViT-B + synthetic data (recommended)": {
        "file": "dinov2_vitb14_c2s1_levir.pth", "type": "dinov2", "preset": "synthetic",
        "tile_size": 252, "overlap": 64},
    "Changen2 ViT-L + DINOv2 synthetic (most accurate, research use)": {
        "file": "dinov2_vitb14_c2s1_levir.pth", "type": "ensemble", "preset": "ensemble",
        "tile_size": 252, "overlap": 64},
    "DINOv2 ViT-B (generalizable)":  {"file": "dinov2_vitb14_levir.pth", "type": "dinov2"},
    "DINOv2 ViT-B (fine-tuned)":     {"file": "dinov2_vitb14_egybcd.pth", "type": "dinov2"},
}

DEFAULT_WEIGHTS = "dinov2_vitb14_c2s1_levir.pth"

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

# The SCD UPerNet's six semantic outputs are not in SECOND_SEMANTIC_CLASSES
# order: measured on the SECOND test set, its output (index + 1) maps to the
# class below (1 low vegetation, 3 tree, 4 water; the rest already match).
# Versions before 0.7 skipped this and labelled those three classes wrongly.
SCD_OUTPUT_TO_SECOND = (0, 3, 2, 4, 1, 5, 6)


# Labelled change (type dinov2_lc): classes of landcover.LandCoverHead after
# merging OpenEarthMap's developed space and road.
LANDCOVER_CLASSES = (
    "no change", "bare ground", "grass / low vegetation", "paved / road",
    "trees", "water", "farmland", "building",
)
LANDCOVER_PALETTE = (
    (255, 255, 255), (210, 180, 140), (144, 238, 144), (128, 128, 128),
    (34, 139, 34), (30, 144, 255), (255, 215, 0), (178, 34, 34),
)


def preset_threshold(preset, tta):
    """The preset's tested threshold for how it is run.

    Presets were tuned with flip averaging; run without it (fast mode), a
    preset may carry its own "fast_threshold", tuned the same way.
    """
    if preset.get("tta") and not tta and "fast_threshold" in preset:
        return float(preset["fast_threshold"])
    return float(preset["threshold"])


def is_scd_model(display_name):
    """Models listed in semantic mode."""
    entry = MODEL_REGISTRY.get(display_name)
    return entry is not None and entry.get("type") in ("opencd_scd", "dinov2_lc")


def is_labelled_model(display_name):
    """Semantic models that write labelled polygons (GeoPackage)."""
    entry = MODEL_REGISTRY.get(display_name)
    return entry is not None and entry.get("type") == "dinov2_lc"


def resolve_weights_path(display_name):
    entry = MODEL_REGISTRY.get(display_name)
    if entry is None:
        raise ValueError(f"Unknown model: {display_name}")
    return os.path.join(_PROJECT_ROOT, entry["file"])
