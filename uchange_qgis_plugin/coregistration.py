import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from . import _gdal_compat  # noqa: F401


@dataclass
class CoregResult:
    success: bool
    corrected_path: Optional[str] = None
    shift_x_px: Optional[float] = None
    shift_y_px: Optional[float] = None
    shift_x_map: Optional[float] = None
    shift_y_map: Optional[float] = None
    message: str = ""
    _temp_path: Optional[str] = field(default=None, repr=False)

    def cleanup(self):
        if self._temp_path and os.path.isfile(self._temp_path):
            try:
                os.remove(self._temp_path)
            except OSError:
                pass
            self._temp_path = None
            self.corrected_path = None


def _check_crs_compatible(ref_path: str, tgt_path: str) -> Optional[str]:
    """Return an error message if CRS are incompatible, None if OK."""
    from osgeo import gdal, osr
    gdal.UseExceptions()
    ref_ds = gdal.Open(ref_path, gdal.GA_ReadOnly)
    tgt_ds = gdal.Open(tgt_path, gdal.GA_ReadOnly)
    if not ref_ds or not tgt_ds:
        return None

    ref_srs = osr.SpatialReference(ref_ds.GetProjection())
    tgt_srs = osr.SpatialReference(tgt_ds.GetProjection())
    ref_ds = tgt_ds = None

    if not ref_srs.GetAuthorityCode(None) and not tgt_srs.GetAuthorityCode(None):
        return None

    if not ref_srs.IsSame(tgt_srs):
        return (
            f"CRS mismatch: reference is {ref_srs.GetName()}, "
            f"target is {tgt_srs.GetName()}. "
            f"Reproject one image to match the other before co-registration."
        )
    return None


_GDAL_DTYPE_RANGE = {
    1: (0, 255),           # GDT_Byte
    2: (0, 65535),         # GDT_UInt16
    3: (-32768, 32767),    # GDT_Int16
    4: (0, 4294967295),    # GDT_UInt32
    5: (-2147483648, 2147483647),  # GDT_Int32
}


def _has_invalid_nodata(path: str) -> bool:
    """Check if any band has a nodata value outside its data type range."""
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if not ds:
        return False
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        nd = band.GetNoDataValue()
        if nd is None:
            continue
        bounds = _GDAL_DTYPE_RANGE.get(band.DataType)
        if bounds and not (bounds[0] <= nd <= bounds[1]):
            ds = None
            return True
    ds = None
    return False


def _strip_nodata_vrt(path: str) -> Optional[str]:
    """If the image has an invalid nodata value, create a VRT wrapper without it.

    Returns the VRT path (caller must delete), or None if no fix was needed.
    """
    if not _has_invalid_nodata(path):
        return None
    from osgeo import gdal
    gdal.UseExceptions()
    fd, vrt_path = tempfile.mkstemp(suffix='.vrt', prefix='coreg_nd_')
    os.close(fd)
    gdal.Translate(vrt_path, path, format='VRT')
    ds = gdal.Open(vrt_path, gdal.GA_Update)
    for i in range(1, ds.RasterCount + 1):
        ds.GetRasterBand(i).DeleteNoDataValue()
    ds.FlushCache()
    ds = None
    return vrt_path


def coregister_images(
    ref_path: str,
    tgt_path: str,
    max_shift: int = 50,
    window_size: tuple = (1024, 1024),
) -> CoregResult:
    """Co-register target image to reference using AROSICS global shift correction.

    Returns CoregResult with success status, corrected file path, and shift info.
    Raises on unexpected errors (callers should catch Exception).
    """
    crs_err = _check_crs_compatible(ref_path, tgt_path)
    if crs_err:
        return CoregResult(success=False, message=crs_err)

    from arosics import COREG

    ref_vrt = _strip_nodata_vrt(ref_path)
    tgt_vrt = _strip_nodata_vrt(tgt_path)
    effective_ref = ref_vrt or ref_path
    effective_tgt = tgt_vrt or tgt_path

    fd, temp_path = tempfile.mkstemp(suffix='.tif', prefix='coreg_')
    os.close(fd)

    try:
        coreg = COREG(
            im_ref=effective_ref,
            im_tgt=effective_tgt,
            path_out=temp_path,
            fmt_out='GTIFF',
            ws=window_size,
            max_shift=max_shift,
            align_grids=True,
            match_gsd=False,
            q=True,
            progress=False,
            ignore_errors=True,
        )

        coreg.calculate_spatial_shifts()

        if not coreg.success:
            os.remove(temp_path)
            return CoregResult(
                success=False,
                message="No reliable shift detected between images",
            )

        if (abs(coreg.x_shift_px or 0) < 0.01
                and abs(coreg.y_shift_px or 0) < 0.01):
            os.remove(temp_path)
            return CoregResult(
                success=True,
                shift_x_px=coreg.x_shift_px,
                shift_y_px=coreg.y_shift_px,
                message="Images are already well-aligned (shift < 0.01px)",
            )

        coreg.correct_shifts()

        return CoregResult(
            success=True,
            corrected_path=temp_path,
            shift_x_px=coreg.x_shift_px,
            shift_y_px=coreg.y_shift_px,
            shift_x_map=getattr(coreg, 'x_shift_map', None),
            shift_y_map=getattr(coreg, 'y_shift_map', None),
            message=(
                f"Corrected shift: "
                f"X={coreg.x_shift_px:.3f}px, Y={coreg.y_shift_px:.3f}px"
            ),
            _temp_path=temp_path,
        )

    except Exception:
        if os.path.isfile(temp_path):
            os.remove(temp_path)
        raise
    finally:
        for vrt in (ref_vrt, tgt_vrt):
            if vrt and os.path.isfile(vrt):
                try:
                    os.remove(vrt)
                except OSError:
                    pass
