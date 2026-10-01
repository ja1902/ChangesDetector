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

    image = np.stack(bands, axis=-1)
    if image.dtype != np.uint8:
        max_val = image.max()
        if max_val > 255:
            image = (image.astype(np.float64) / max_val * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
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


def polygonize_mask(binary_mask, geotransform, projection_wkt, output_path,
                    min_area=0, style='exact'):
    """Convert a binary mask to vector polygons and save as GeoPackage.

    Args:
        binary_mask: 2D uint8 numpy array (H, W), 1=change, 0=no-change
        geotransform: GDAL geotransform from the input raster
        projection_wkt: WKT projection string
        output_path: path to output .gpkg file
        min_area: minimum polygon area in map units; smaller polygons are removed
        style: 'exact', 'simplified', or 'convex hull'
    """
    binary_mask = _smooth_mask(binary_mask)

    h, w = binary_mask.shape

    mem_driver = gdal.GetDriverByName("MEM")
    mem_ds = mem_driver.Create("", w, h, 1, gdal.GDT_Byte)
    mem_ds.SetGeoTransform(geotransform)
    mem_ds.SetProjection(projection_wkt)
    mem_band = mem_ds.GetRasterBand(1)
    write_band(mem_band, binary_mask.astype(np.uint8))
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
    field_defn = ogr.FieldDefn("change", ogr.OFTInteger)
    layer.CreateField(field_defn)

    gdal.Polygonize(mem_band, mem_band, layer, 0, [], callback=None)

    total_features = layer.GetFeatureCount()
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
        geom = _remove_small_holes(geom, min_area)
        styled = _apply_style(geom, style, pixel_w)
        feat.SetGeometry(styled)
        layer.SetFeature(feat)

    for fid in to_delete:
        layer.DeleteFeature(fid)

    final_features = layer.GetFeatureCount()

    out_ds.FlushCache()
    out_ds = None
    mem_ds = None

    return total_features, final_features


def polygonize_labelled_changes(blobs, from_cls, to_cls, class_names, geotransform,
                                projection_wkt, output_path, min_area=0, style='exact'):
    """Change blobs -> GeoPackage polygons carrying their before/after class.

    Args:
        blobs: 2D int32 array of blob ids (0 = no change)
        from_cls, to_cls: per-blob class ids (index = blob id)
        class_names: names for the class ids
        min_area: minimum polygon area in map units
    Fields: from_class, to_class, change ("farmland -> building"), area.
    """
    h, w = blobs.shape
    mem_ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Int32)
    mem_ds.SetGeoTransform(geotransform)
    mem_ds.SetProjection(projection_wkt)
    mem_band = mem_ds.GetRasterBand(1)
    write_band(mem_band, blobs.astype(np.int32))
    mem_band.FlushCache()

    out_ds = ogr.GetDriverByName("GPKG").CreateDataSource(output_path)
    srs = None
    if projection_wkt:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection_wkt)
    layer = out_ds.CreateLayer("changes", srs=srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("blob", ogr.OFTInteger))
    for name in ("from_class", "to_class", "change"):
        layer.CreateField(ogr.FieldDefn(name, ogr.OFTString))
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
        k = feat.GetField("blob")
        a, b = class_names[from_cls[k]], class_names[to_cls[k]]
        geom = _apply_style(_remove_small_holes(geom, min_area), style, pixel_w)
        feat.SetGeometry(geom)
        feat.SetField("from_class", a)
        feat.SetField("to_class", b)
        feat.SetField("change", "%s -> %s" % (a, b))
        feat.SetField("area", geom.GetArea())
        layer.SetFeature(feat)
    for fid in to_delete:
        layer.DeleteFeature(fid)
    final = layer.GetFeatureCount()
    out_ds.FlushCache()
    out_ds = None
    mem_ds = None
    return total, final
