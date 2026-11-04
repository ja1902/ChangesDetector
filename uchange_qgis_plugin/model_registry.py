import os

_PLUGIN_DIR = os.path.realpath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_PLUGIN_DIR, ".."))

# Inference presets: the exact pipeline each model was evaluated with in
# research/RESULTS_2026-09.md, so its recommended threshold means the same
# thing here. Thresholds are on logit-adjusted probabilities and were chosen on
# validation data from unfamiliar imagery (e.g. S2Looking val), not LEVIR.
#   scale         upsampling applied before tiling (DINOv2 was run at 1.4x)
#   tta           average the four flips
#   logit_adjust  remove the LEVIR change prior (pi = 0.0584) from the logits
#   target_gsd    resample coarser imagery up to this pixel size (metres)
#   tta_gate      flip-average only tiles reaching this single-pass probability
#   label_free    weight of the label-free change score nudge (labelfree.py), 0 = off
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
        # Flip averaging only on tiles whose single pass reaches 0.5 somewhere:
        # same masks as averaging every tile on S2Looking and LEVIR
        # (0.001% of pixels differ), 38% less inference time on GPU
        # (benchmarks/bench.py).
        "tta_gate": 0.5,
    },
    # The finetuned model: as "synthetic", plus the false-alarm filter, on by
    # default (falsealarm_filter.py, --no-filter-false-alarms to turn it off). Blobs found
    # at "low", kept if the classifier's real probability >= "min_score", drawn at
    # "redraw". Chosen on 8 hand-checked Google Earth squares, each scored by a
    # model trained without its place: 69% of changes found with 10 false alarms
    # per km2. Checked unchanged on pr2_c1 (23 changes, unseen): 16/23 with 14
    # false alarms per km2 (work_dirs/exp_recheck).
    # The label-free nudge (labelfree.py) is no longer used: with the filter it
    # adds nothing, alone it is level at 20 false alarms per km2 and worse below.
    # "threshold" (0.85, best object F1 without the filter) applies when the
    # filter is off.
    "finetuned": {
        "tile_size": 252, "overlap": 64, "scale": 1.4, "tta": True,
        "logit_adjust": True, "target_gsd": 0.5, "threshold": 0.85,
        "core_threshold": 0.0, "min_blob_m2": 0.0, "min_width_m": 0.0,
        "fast_mode": True, "tta_gate": 0.5,
        "false_alarm_filter": {"file": "weights/falsealarm_filter_finetuned.npz",
                               "low": 0.6, "redraw": 0.83, "min_score": 0.46},
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
        "file": "dinov2_vitb14_finetuned.pth", "type": "dinov2_lc", "preset": "finetuned",
        "landcover_head": "landcover_dinov2_vitb14_oem_second.pth",
        "tile_size": 252, "overlap": 64},
    # Binary mode: the finetuned model, and the synthetic-data model for new
    # buildings only. The Changen2 ensemble and the v0.6 models are no longer
    # offered; the CLI still runs the ensemble for research (--model-type ensemble).
    "DINOv2 ViT-B finetuned (recommended)": {
        "file": "dinov2_vitb14_finetuned.pth", "type": "dinov2", "preset": "finetuned",
        "tile_size": 252, "overlap": 64},
    "DINOv2 ViT-B + synthetic data (general)": {
        "file": "dinov2_vitb14_c2s1_levir.pth", "type": "dinov2", "preset": "synthetic",
        "tile_size": 252, "overlap": 64},
}

# CLI defaults: both modes run the finetuned model (the land-cover head reads only the
# frozen backbone, the same in every DINOv2 checkpoint). DEFAULT_WEIGHTS is the
# synthetic model, used when a preset other than "finetuned" is given without weights.
DEFAULT_WEIGHTS = "dinov2_vitb14_c2s1_levir.pth"
FINETUNED_WEIGHTS = "dinov2_vitb14_finetuned.pth"

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


def preset_threshold(preset, tta, label_free=None):
    """The preset's tested threshold for how it is run.

    Presets were tuned with flip averaging; run without it (fast mode), a
    preset may carry its own "fast_threshold", tuned the same way. A preset
    with a label-free nudge carries "threshold_no_label_free" for when the
    nudge is turned off (label_free == 0).
    """
    if preset.get("label_free") and label_free == 0 and "threshold_no_label_free" in preset:
        return float(preset["threshold_no_label_free"])
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
