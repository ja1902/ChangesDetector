import numpy as np

from osgeo import gdal, ogr, osr

gdal.UseExceptions()

# Band I/O through raw buffers instead of ReadAsArray / WriteArray: those need
# GDAL's optional NumPy module (osgeo._gdal_array), which is missing when the
# bindings were built without NumPy, and then every GeoTIFF read and polygon
# write fails.
_GDAL_TO_NUMPY = {
    gdal.GDT_Byte: np.uint8, gdal.GDT_UInt16: np.uint16, gdal.GDT_Int16: np.int16,
    gdal.GDT_UInt32: np.uint32, gdal.GDT_Int32: np.int32,
    gdal.GDT_Float32: np.float32, gdal.GDT_Float64: np.float64,
}
_NUMPY_TO_GDAL = {np.dtype(v): k for k, v in _GDAL_TO_NUMPY.items()}


def read_band(band):
    """GDAL band -> 2D numpy array (no osgeo.gdal_array needed)."""
    dt = band.DataType if band.DataType in _GDAL_TO_NUMPY else gdal.GDT_Float64
    w, h = band.XSize, band.YSize
    buf = band.ReadRaster(0, 0, w, h, buf_type=dt)
    return np.frombuffer(buf, dtype=_GDAL_TO_NUMPY[dt]).reshape(h, w)


def write_band(band, array):
    """2D numpy array -> GDAL band (no osgeo.gdal_array needed)."""
    arr = np.ascontiguousarray(array)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    if arr.dtype not in _NUMPY_TO_GDAL:
        arr = arr.astype(np.float64)
    h, w = arr.shape
    band.WriteRaster(0, 0, w, h, arr.tobytes(), w, h, _NUMPY_TO_GDAL[arr.dtype])


def bands_to_rgb8(bands):
    """2D bands -> (H, W, n) uint8; values above 255 are scaled by the maximum.

    Same values as (np.stack(bands, -1).astype(np.float64) / max * 255).astype(np.uint8),
    converted a block of rows at a time, so a 16-bit image never needs a
    float64 copy of the whole raster (8x its 8-bit size).
    """
    h, w = bands[0].shape
    out = np.empty((h, w, len(bands)), dtype=np.uint8)
    max_val = None
    if any(b.dtype != np.uint8 for b in bands):
        max_val = np.max([b.max() for b in bands])
    step = max(1, (1 << 20) // max(w, 1))
    for i, band in enumerate(bands):
        for y in range(0, h, step):
            blk = band[y:y + step]
            if max_val is not None and max_val > 255:
                blk = blk.astype(np.float64) / max_val * 255
            out[y:y + step, :, i] = blk.astype(np.uint8)
    return out


def read_raster(source_path):
    """Read a raster file and return image data with georeferencing info.

    Returns:
        image: numpy array (H, W, 3) uint8 RGB
        geotransform: 6-element GDAL geotransform tuple
        projection_wkt: WKT string of the spatial reference
    """
    ds = gdal.Open(source_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {source_path}")

    band_count = ds.RasterCount
    if band_count < 3:
        raise ValueError(
            f"Raster must have at least 3 bands (RGB), got {band_count}"
        )

    bands = []
    for i in range(1, 4):
        band = read_band(ds.GetRasterBand(i))
        bands.append(band)

    image = bands_to_rgb8(bands)
    geotransform = ds.GetGeoTransform()
    projection_wkt = ds.GetProjection()

    ds = None
    return image, geotransform, projection_wkt


def _smooth_mask(binary_mask, kernel_size=5):
    """Morphological close then open to smooth jagged mask edges."""
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    smoothed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    return smoothed


def _apply_style(geom, style, pixel_w):
    """Apply output style to a geometry.

    Styles:
        exact: mild smoothing only (pixel_w * 2)
        simplified: stronger Douglas-Peucker (pixel_w * 6)
        convex hull: convex hull of the geometry
    """
    if style == 'convex hull':
        hull = geom.ConvexHull()
        return hull if hull and not hull.IsEmpty() else geom
    elif style == 'simplified':
        simplified = geom.Simplify(pixel_w * 6)
        return simplified if simplified and not simplified.IsEmpty() else geom
    else:
        simplified = geom.Simplify(pixel_w * 2)
        return simplified if simplified and not simplified.IsEmpty() else geom


def save_mask_png(binary_mask, output_path):
    """Save binary mask as a grayscale PNG (255=change, 0=no-change)."""
    import cv2
    cv2.imwrite(output_path, (binary_mask * 255).astype(np.uint8),
                [cv2.IMWRITE_PNG_COMPRESSION, 1])


def save_semantic_geotiff(class_map, geotransform, projection_wkt, output_path,
                          class_names, class_colors):
    """Save a semantic class map as a GeoTIFF with a GDAL color table.

    Args:
        class_map: 2D uint8 array (H, W) with class indices
        geotransform: GDAL geotransform tuple
        projection_wkt: WKT projection string
        output_path: path to output .tif file
        class_names: tuple of class name strings
        class_colors: tuple of (R, G, B) tuples, one per class
    """
    h, w = class_map.shape

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(output_path, w, h, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(geotransform)
    if projection_wkt:
        ds.SetProjection(projection_wkt)

    band = ds.GetRasterBand(1)

    ct = gdal.ColorTable()
    for i, color in enumerate(class_colors):
        alpha = 0 if i == 0 else 255
        ct.SetColorEntry(i, (*color, alpha))
    band.SetColorTable(ct)
    band.SetColorInterpretation(gdal.GCI_PaletteIndex)
    band.SetNoDataValue(0)

    write_band(band, class_map.astype(np.uint8))
    band.SetCategoryNames(list(class_names))
    band.FlushCache()
    ds.FlushCache()
    ds = None


def save_binary_geotiff(binary_mask, geotransform, projection_wkt, output_path):
    """Save a binary change mask as a single-band GeoTIFF (255=change, 0=no-change)."""
    h, w = binary_mask.shape

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(output_path, w, h, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(geotransform)
    if projection_wkt:
        ds.SetProjection(projection_wkt)

    band = ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    write_band(band, (binary_mask * 255).astype(np.uint8))
    band.FlushCache()
    ds.FlushCache()
    ds = None


def _remove_small_holes(geom, min_area):
    """Remove interior rings (holes) smaller than min_area from a polygon."""
    if min_area <= 0:
        return geom
    geom_type = geom.GetGeometryType()
    if geom_type == ogr.wkbPolygon:
        if geom.GetGeometryCount() <= 1:
            return geom
        exterior = geom.GetGeometryRef(0).Clone()
        new_poly = ogr.Geometry(ogr.wkbPolygon)
        new_poly.AddGeometry(exterior)
        for i in range(1, geom.GetGeometryCount()):
            ring = geom.GetGeometryRef(i)
            hole = ogr.Geometry(ogr.wkbPolygon)
            hole.AddGeometry(ring.Clone())
            if abs(hole.GetArea()) >= min_area:
                new_poly.AddGeometry(ring.Clone())
        return new_poly
    elif geom_type == ogr.wkbMultiPolygon:
        new_multi = ogr.Geometry(ogr.wkbMultiPolygon)
        for i in range(geom.GetGeometryCount()):
            new_multi.AddGeometry(_remove_small_holes(geom.GetGeometryRef(i), min_area))
        return new_multi
    return geom


def change_scores(ids, n, prob=None, filter_labels=None, filter_real=None):
    """Per-change scores (index = change id, 0 unused) for the polygon attributes.

    confidence    mean change probability inside the change (prob: (H, W) map)
    filter_score  the false-alarm filter's probability that the change is real, from the
                  filter blob it was drawn from (filter_labels: blob id per pixel,
                  filter_real: probability per blob id); averaged over its pixels if
                  smoothing joined two blobs
    Missing inputs give None for that score.
    """
    m = ids > 0
    k = ids[m]
    cnt = np.bincount(k, minlength=n).astype(np.float64)
    conf = real = None
    if prob is not None:
        conf = np.bincount(k, weights=prob[m], minlength=n) / np.maximum(cnt, 1)
    if filter_labels is not None and filter_real is not None:
        lab = filter_labels[m]
        r = np.asarray(filter_real, np.float64)[lab]
        ok = (lab > 0) & np.isfinite(r)
        n_ok = np.bincount(k[ok], minlength=n)
        real = np.bincount(k[ok], weights=r[ok], minlength=n) / np.maximum(n_ok, 1)
        real[n_ok == 0] = np.nan
    return conf, real


def polygonize_changes(ids, geotransform, projection_wkt, output_path, fields=(),
                       min_area=0, style='exact'):
    """Change ids -> GeoPackage polygons, one per change (8-connected), with attributes.

    Args:
        ids: 2D int32 array of change ids (0 = no change)
        fields: (name, kind, values) with kind "int", "real" or "str" and values indexed by
            change id; NaN / None values are left empty
        min_area: minimum polygon area in map units; smaller polygons are removed
        style: 'exact', 'simplified', or 'convex hull'
    Always writes "id" (the change id) and "area" (map units^2, after styling).
    Returns (polygons created, polygons kept).
    """
    h, w = ids.shape
    mem_ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Int32)
    mem_ds.SetGeoTransform(geotransform)
    mem_ds.SetProjection(projection_wkt)
    mem_band = mem_ds.GetRasterBand(1)
    write_band(mem_band, ids.astype(np.int32))
    mem_band.FlushCache()

    gpkg_driver = ogr.GetDriverByName("GPKG")
    if gpkg_driver is None:
        raise RuntimeError("GPKG OGR driver not available")
    out_ds = gpkg_driver.CreateDataSource(output_path)
    srs = None
    if projection_wkt:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection_wkt)
    layer = out_ds.CreateLayer("changes", srs=srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    kinds = {"int": ogr.OFTInteger, "real": ogr.OFTReal, "str": ogr.OFTString}
    for name, kind, _ in fields:
        layer.CreateField(ogr.FieldDefn(name, kinds[kind]))
    layer.CreateField(ogr.FieldDefn("area", ogr.OFTReal))

    gdal.Polygonize(mem_band, mem_band, layer, 0, ["8CONNECTED=8"], callback=None)
    total = layer.GetFeatureCount()
    pixel_w = abs(geotransform[1])

    layer.ResetReading()
    to_delete = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        if min_area > 0 and geom.GetArea() < min_area:
            to_delete.append(feat.GetFID())
            continue
        k = feat.GetField("id")
        geom = _apply_style(_remove_small_holes(geom, min_area), style, pixel_w)
        feat.SetGeometry(geom)
        for name, kind, values in fields:
            v = values[k]
            if v is None or (kind == "real" and not np.isfinite(v)):
                continue
            feat.SetField(name, int(v) if kind == "int" else float(v) if kind == "real" else str(v))
        feat.SetField("area", geom.GetArea())
        layer.SetFeature(feat)
    for fid in to_delete:
        layer.DeleteFeature(fid)
    final = layer.GetFeatureCount()
    out_ds.FlushCache()
    out_ds = None
    mem_ds = None
    return total, final


def _score_fields(n, prob, filter_labels, filter_real, ids):
    conf, real = change_scores(ids, n, prob, filter_labels, filter_real)
    fields = []
    if conf is not None:
        fields.append(("confidence", "real", np.round(conf, 3)))
    if real is not None:
        fields.append(("filter_score", "real", np.round(real, 3)))
    return fields


def polygonize_mask(binary_mask, geotransform, projection_wkt, output_path,
                    min_area=0, style='exact', prob=None, filter_labels=None, filter_real=None):
    """Convert a binary mask to vector polygons and save as GeoPackage.

    Args:
        binary_mask: 2D uint8 numpy array (H, W), 1=change, 0=no-change
        geotransform: GDAL geotransform from the input raster
        projection_wkt: WKT projection string
        output_path: path to output .gpkg file
        min_area: minimum polygon area in map units; smaller polygons are removed
        style: 'exact', 'simplified', or 'convex hull'
        prob, filter_labels, filter_real: optional, for the "confidence" and
            "filter_score" fields (see change_scores)
    Fields: id, change (1), confidence, filter_score (with the filter), area.
    """
    import cv2
    n, ids = cv2.connectedComponents(_smooth_mask(binary_mask.astype(np.uint8)), connectivity=8)
    fields = [("change", "int", np.ones(n, np.int64))]
    fields += _score_fields(n, prob, filter_labels, filter_real, ids)
    return polygonize_changes(ids, geotransform, projection_wkt, output_path, fields,
                              min_area=min_area, style=style)


def change_name(from_name, to_name):
    """The "change" field of a labelled polygon."""
    if from_name == to_name:
        return "%s (class unchanged)" % from_name
    return "%s -> %s" % (from_name, to_name)


def polygonize_labelled_changes(blobs, from_cls, to_cls, class_names, geotransform,
                                projection_wkt, output_path, min_area=0, style='exact',
                                prob=None, filter_labels=None, filter_real=None):
    """Change blobs -> GeoPackage polygons carrying their before/after class.

    Args:
        blobs: 2D int32 array of change ids (0 = no change)
        from_cls, to_cls: per-change class ids (index = change id)
        class_names: names for the class ids
        min_area: minimum polygon area in map units
        prob, filter_labels, filter_real: optional, as in polygonize_mask
    Fields: id, from_class, to_class, change ("farmland -> building", or "paved / road
    (class unchanged)" when the land cover is the same on both dates: a detected change
    the classes can't name, such as a flat roof read as paving), confidence,
    filter_score (with the filter), area.
    """
    n = len(from_cls)
    a = [class_names[c] for c in from_cls]
    b = [class_names[c] for c in to_cls]
    fields = [("from_class", "str", a), ("to_class", "str", b),
              ("change", "str", [change_name(x, y) for x, y in zip(a, b)])]
    fields += _score_fields(n, prob, filter_labels, filter_real, blobs)
    return polygonize_changes(blobs, geotransform, projection_wkt, output_path, fields,
                              min_area=min_area, style=style)
