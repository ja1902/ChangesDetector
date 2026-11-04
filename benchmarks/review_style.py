"""Default QGIS styles for the review layers, stored inside candidates.gpkg.

QGIS applies a style from a GeoPackage's layer_styles table when the layer is
opened: candidates are outlined by verdict (yellow = to review, green = real,
red = false, grey = unsure) and verdict / type / direction / confidence get
drop-down lists instead of free text.
"""
import sqlite3

VERDICTS = ["real", "false", "unsure"]
TYPES = ["building", "road", "car_park", "construction_site", "demolition", "other"]
DIRECTIONS = ["built", "demolished"]
CONFIDENCE = ["sure", "unsure"]


def _value_map(field, values):
    items = "".join(f'<Option type="Map"><Option name="{v}" type="QString" value="{v}"/></Option>'
                    for v in values)
    return (f'<field name="{field}" configurationFlags="None"><editWidget type="ValueMap"><config>'
            f'<Option type="Map"><Option name="map" type="List">{items}</Option></Option>'
            f'</config></editWidget></field>')


def _fill(name, rgb, width):
    return (f'<symbol type="fill" name="{name}" alpha="1" clip_to_extent="1" force_rhr="0">'
            f'<layer class="SimpleFill" enabled="1" locked="0" pass="0"><Option type="Map">'
            f'<Option name="color" type="QString" value="0,0,0,0"/>'
            f'<Option name="style" type="QString" value="no"/>'
            f'<Option name="outline_color" type="QString" value="{rgb},255"/>'
            f'<Option name="outline_style" type="QString" value="solid"/>'
            f'<Option name="outline_width" type="QString" value="{width}"/>'
            f'<Option name="outline_width_unit" type="QString" value="MM"/>'
            f'</Option></layer></symbol>')


def candidates_qml():
    cats = [("", "to review", "255,221,0", 0.7), ("real", "real", "0,200,70", 0.7),
            ("false", "false", "230,40,40", 0.5), ("unsure", "unsure", "150,150,150", 0.5)]
    categories = "".join(f'<category symbol="{i}" value="{v}" label="{lab}" render="true"/>'
                         for i, (v, lab, _, _) in enumerate(cats))
    symbols = "".join(_fill(str(i), rgb, w) for i, (_, _, rgb, w) in enumerate(cats))
    fields = _value_map("verdict", VERDICTS) + _value_map("type", TYPES)
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>"
            '<qgis version="3.22" styleCategories="Symbology|Fields|Forms">'
            '<renderer-v2 type="categorizedSymbol" attr="verdict" symbollevels="0" '
            'enableorderby="0" forceraster="0">'
            f'<categories>{categories}</categories><symbols>{symbols}</symbols></renderer-v2>'
            f'<fieldConfiguration>{fields}</fieldConfiguration></qgis>')


def missed_qml():
    marker = ('<symbol type="marker" name="0" alpha="1" clip_to_extent="1" force_rhr="0">'
              '<layer class="SimpleMarker" enabled="1" locked="0" pass="0"><Option type="Map">'
              '<Option name="name" type="QString" value="cross2"/>'
              '<Option name="color" type="QString" value="0,170,255,255"/>'
              '<Option name="outline_color" type="QString" value="0,170,255,255"/>'
              '<Option name="outline_width" type="QString" value="0.8"/>'
              '<Option name="size" type="QString" value="4"/>'
              '</Option></layer></symbol>')
    fields = (_value_map("type", TYPES) + _value_map("direction", DIRECTIONS)
              + _value_map("confidence", CONFIDENCE))
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>"
            '<qgis version="3.22" styleCategories="Symbology|Fields|Forms">'
            f'<renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0" forceraster="0">'
            f'<symbols>{marker}</symbols></renderer-v2>'
            f'<fieldConfiguration>{fields}</fieldConfiguration></qgis>')


def missed_areas_qml():
    symbol = _fill("0", "0,170,255", 0.8)
    fields = _value_map("type", TYPES) + _value_map("direction", DIRECTIONS)
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>"
            '<qgis version="3.22" styleCategories="Symbology|Fields|Forms">'
            '<renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0" forceraster="0">'
            f'<symbols>{symbol}</symbols></renderer-v2>'
            f'<fieldConfiguration>{fields}</fieldConfiguration></qgis>')


def add_styles(gpkg, tables=None):
    """tables: {table name: QML}; default: the candidates.gpkg layers."""
    if tables is None:
        tables = {"candidates": candidates_qml(), "missed": missed_qml()}
    con = sqlite3.connect(str(gpkg))
    con.execute("""CREATE TABLE IF NOT EXISTS layer_styles (
        id INTEGER PRIMARY KEY AUTOINCREMENT, f_table_catalog TEXT(256), f_table_schema TEXT(256),
        f_table_name TEXT(256), f_geometry_column TEXT(256), styleName TEXT(30), styleQML TEXT,
        styleSLD TEXT, useAsDefault BOOLEAN, description TEXT, owner TEXT(30), ui TEXT(30),
        update_time DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    for table, qml in tables.items():
        con.execute("DELETE FROM layer_styles WHERE f_table_name = ?", (table,))
        con.execute("INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name, "
                    "f_geometry_column, styleName, styleQML, styleSLD, useAsDefault, description) "
                    "VALUES ('', '', ?, 'geom', 'review', ?, '', 1, 'review style')", (table, qml))
    con.commit()
    con.close()


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        add_styles(path)
        print(f"styled {path}")


def grid_qml():
    def fill(name, fill_rgba, outline, style):
        return (f'<symbol type="fill" name="{name}" alpha="1" clip_to_extent="1" force_rhr="0">'
                f'<layer class="SimpleFill" enabled="1" locked="0" pass="0"><Option type="Map">'
                f'<Option name="color" type="QString" value="{fill_rgba}"/>'
                f'<Option name="style" type="QString" value="solid"/>'
                f'<Option name="outline_color" type="QString" value="{outline}"/>'
                f'<Option name="outline_style" type="QString" value="{style}"/>'
                f'<Option name="outline_width" type="QString" value="0.6"/>'
                f'<Option name="outline_width_unit" type="QString" value="MM"/>'
                f'</Option></layer></symbol>')
    symbols = (fill("0", "0,0,0,0", "255,255,255,255", "dash")
               + fill("1", "40,40,40,150", "255,255,255,255", "solid"))
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>"
            '<qgis version="3.22" styleCategories="Symbology|Fields|Forms">'
            '<renderer-v2 type="categorizedSymbol" attr="done" symbollevels="0" enableorderby="0" forceraster="0">'
            '<categories><category symbol="0" value="" label="to check" render="true"/>'
            '<category symbol="1" value="yes" label="done" render="true"/></categories>'
            f'<symbols>{symbols}</symbols></renderer-v2>'
            f'<fieldConfiguration>{_value_map("done", ["yes"])}</fieldConfiguration></qgis>')


def make_grid(square_dir, n=4):
    """grid.gpkg: n x n cells over the square, with a 'done' field, styled for review."""
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
    ds = gdal.Open(str(square_dir / "before.tif"))
    gt, w, h = ds.GetGeoTransform(), ds.RasterXSize, ds.RasterYSize
    path = square_dir / "grid.gpkg"
    if path.exists():
        path.unlink()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    out = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    lyr = out.CreateLayer("grid", srs, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("cell", ogr.OFTInteger))
    lyr.CreateField(ogr.FieldDefn("done", ogr.OFTString))
    k = 0
    for r in range(n):
        for c in range(n):
            k += 1
            x0, x1 = gt[0] + gt[1] * w * c / n, gt[0] + gt[1] * w * (c + 1) / n
            y0, y1 = gt[3] + gt[5] * h * r / n, gt[3] + gt[5] * h * (r + 1) / n
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)):
                ring.AddPoint_2D(x, y)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            f = ogr.Feature(lyr.GetLayerDefn())
            f.SetGeometry(poly)
            f.SetField("cell", k)
            lyr.CreateFeature(f)
    out = None
    con = sqlite3.connect(str(path))
    con.execute("""CREATE TABLE IF NOT EXISTS layer_styles (
        id INTEGER PRIMARY KEY AUTOINCREMENT, f_table_catalog TEXT(256), f_table_schema TEXT(256),
        f_table_name TEXT(256), f_geometry_column TEXT(256), styleName TEXT(30), styleQML TEXT,
        styleSLD TEXT, useAsDefault BOOLEAN, description TEXT, owner TEXT(30), ui TEXT(30),
        update_time DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    con.execute("INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name, f_geometry_column, "
                "styleName, styleQML, styleSLD, useAsDefault, description) "
                "VALUES ('', '', 'grid', 'geom', 'review', ?, '', 1, 'review grid')", (grid_qml(),))
    con.commit()
    con.close()
    return path
