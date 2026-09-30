"""Speckle removal for binary change masks.

On slightly off-nadir imagery the model finds real changes well but also
leaves many small, lukewarm blobs (building lean, shadows, vegetation). A blob
of the thresholded mask is kept only if it is big enough and contains a core
that is both confident and not a thin sliver. Kept blobs keep their full
extent, so real changes are not eroded.
"""
import numpy as np


def clean_mask(prob, threshold, core=None, min_area_px=0, min_width_px=0):
    """Binary mask (uint8) from a probability map, with speckle removed.

    prob: float (H, W) change probabilities.
    threshold: pixels above it are change.
    core: a blob must contain a pixel at or above this probability (None = off).
    min_area_px: blobs smaller than this many pixels are dropped.
    min_width_px: a blob's core must survive an opening of this width, which
        drops thin slivers along building edges (0 or 1 = off).
    """
    mask = (prob > threshold).astype(np.uint8)
    use_core = core is not None and core > threshold
    width = int(round(min_width_px))
    if not mask.any() or (not use_core and min_area_px <= 1 and width < 2):
        return mask

    import cv2
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= max(1, min_area_px)
    keep[0] = False
    if use_core or width >= 2:
        seeds = prob >= core if use_core else mask.astype(bool)
        if width >= 2:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
            seeds = seeds & cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel).astype(bool)
        has_seed = np.zeros(n, dtype=bool)
        has_seed[np.unique(labels[seeds])] = True
        keep &= has_seed
    return keep[labels].astype(np.uint8)


def to_pixels(gsd_m, min_area_m2=0.0, min_width_m=0.0):
    """Convert the metric clean-up sizes to pixels at a given pixel size."""
    if not gsd_m:
        gsd_m = 0.5                     # the models' training resolution
    return min_area_m2 / (gsd_m * gsd_m), min_width_m / gsd_m
