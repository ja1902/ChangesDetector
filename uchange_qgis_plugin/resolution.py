"""Resolution handling: bring imagery to the scale the model was trained at.

Every model tested collapses on imagery much coarser than its training data
(LEVIR IoU at 2 m: 0.10-0.12), while resampling the image up to about 0.5 m
before inference restores most of it (0.50-0.70). Finer imagery is left alone:
downsampling it did not help consistently. See research/RESULTS_2026-09.md.
"""
import math

import numpy as np
from PIL import Image

# Keep the resampled raster within reach of ordinary GPU/CPU memory.
MAX_SIDE = 24000


def pixel_size_m(geo_info):
    """Mean pixel size in metres from a GDAL geotransform + WKT, or None."""
    if not geo_info or not geo_info.get("projection"):
        return None
    try:
        from osgeo import osr
    except ImportError:
        return None
    gt = geo_info["geotransform"]
    srs = osr.SpatialReference()
    if srs.ImportFromWkt(geo_info["projection"]) != 0:
        return None
    px, py = abs(gt[1]), abs(gt[5])
    if srs.IsProjected():
        unit = srs.GetLinearUnits() or 1.0
        return 0.5 * (px + py) * unit
    if srs.IsGeographic():
        lat = gt[3] + gt[5] * geo_info.get("height", 0) / 2.0
        mx = px * 111320.0 * math.cos(math.radians(lat))
        my = py * 110540.0
        return 0.5 * (mx + my)
    return None


def working_factor(geo_info, target_gsd, scale=1.0, shape=None):
    """Resampling factor for inference: `scale` times the GSD correction.

    Coarser-than-target imagery is upsampled to target_gsd; finer imagery keeps
    its native resolution. Returns (factor, pixel_size_m or None, note).
    """
    gsd = pixel_size_m(geo_info) if target_gsd else None
    gsd_factor = 1.0
    note = ""
    if gsd and target_gsd and gsd > target_gsd * 1.05:
        gsd_factor = gsd / target_gsd
        note = f"pixel size {gsd:.2f} m > {target_gsd} m: upsampling x{gsd_factor:.2f}"
    factor = scale * gsd_factor
    if shape is not None and factor > 1.0:
        limit = MAX_SIDE / float(max(shape[:2]))
        if factor > limit:
            note += (f"; capped x{factor:.2f} -> x{limit:.2f} to fit memory "
                     "(process a smaller extent for the full effect)")
            factor = max(1.0, limit)
    return factor, gsd, note


def resample_rgb(img, factor):
    if abs(factor - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    size = (max(1, int(round(w * factor))), max(1, int(round(h * factor))))
    mode = Image.BILINEAR if factor > 1 else Image.BOX
    return np.array(Image.fromarray(img).resize(size, mode))


def resample_rows(rows, full_hw, out_hw, r0, r1, mode, offset=0):
    """Rows r0:r1 of a full_hw image resized to out_hw, without the whole image.

    `rows` holds rows offset .. offset + len(rows) of the full image; they must
    cover the filter's reach around the requested rows. Matches resizing the
    whole image and slicing, up to rounding (PIL is given the same sampling grid
    through `box`; uint8 output can differ by 1 on a few pixels).
    """
    h, w = full_hw
    s = h / float(out_hw[0])
    c0, c1 = first_row_needed(h, out_hw[0], r0), rows_needed(h, out_hw[0], r1)
    if c0 < offset or c1 > offset + len(rows):
        raise ValueError("resample_rows: rows %d-%d needed, %d-%d given"
                         % (c0, c1, offset, offset + len(rows)))
    part = Image.fromarray(np.ascontiguousarray(rows[c0 - offset:c1 - offset]))
    box = (0, r0 * s - c0, w, r1 * s - c0)
    return np.array(part.resize((out_hw[1], r1 - r0), mode, box=box))


def rows_needed(full_h, out_h, r1):
    """Last input row (exclusive) that resample_rows reads for output rows < r1."""
    s = full_h / float(out_h)
    return min(full_h, int(math.ceil(r1 * s)) + int(math.ceil(2 * max(1.0, s))) + 2)


def first_row_needed(full_h, out_h, r0):
    """First input row that resample_rows reads for output rows >= r0."""
    s = full_h / float(out_h)
    return max(0, int(math.floor(r0 * s)) - int(math.ceil(2 * max(1.0, s))) - 2)


def resample_prob(prob, shape_hw):
    """Float probability map back to (h, w)."""
    h, w = shape_hw
    if prob.shape == (h, w):
        return prob.astype(np.float32)
    return np.array(Image.fromarray(prob.astype(np.float32))
                    .resize((w, h), Image.BILINEAR), dtype=np.float32)
