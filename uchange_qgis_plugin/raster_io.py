import numpy as np

from osgeo import gdal, ogr, osr

gdal.UseExceptions()


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
        band = ds.GetRasterBand(i).ReadAsArray()
        bands.append(band)

    image = np.stack(bands, axis=-1).astype(np.uint8)
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
    mem_band.WriteArray(binary_mask)
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
