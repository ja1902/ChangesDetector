"""GDAL < 3.8 compatibility shims for geoarray/arosics/py_tools_ds.

Adds context manager support to gdal.Dataset, ogr.DataSource, ogr.Layer,
ogr.Feature, and provides a gdal.config_options polyfill.

Must be imported before any code that uses these features (e.g. via __init__.py).
"""
from contextlib import contextmanager
from osgeo import gdal, ogr


def _enter(self):
    return self


def _exit(self, *args):
    return None


def _exit_flush(self, *args):
    self.FlushCache()
    return None


def _exit_destroy(self, *args):
    self.Destroy()
    return None


for _cls in (gdal.Dataset,):
    if not hasattr(_cls, '__enter__'):
        _cls.__enter__ = _enter
        _cls.__exit__ = _exit_flush

for _cls in (ogr.DataSource,):
    if not hasattr(_cls, '__enter__'):
        _cls.__enter__ = _enter
        _cls.__exit__ = _exit_destroy

for _cls_name in ('Layer', 'Feature'):
    _cls = getattr(ogr, _cls_name, None)
    if _cls and not hasattr(_cls, '__enter__'):
        _cls.__enter__ = _enter
        _cls.__exit__ = _exit

if not hasattr(gdal, 'config_options'):
    @contextmanager
    def _config_options(options):
        old = {}
        for k, v in options.items():
            old[k] = gdal.GetConfigOption(k)
            gdal.SetConfigOption(k, v)
        try:
            yield
        finally:
            for k, v in old.items():
                gdal.SetConfigOption(k, v)
    gdal.config_options = _config_options
