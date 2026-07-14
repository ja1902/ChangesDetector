from . import _gdal_compat  # noqa: F401 — must run before geoarray/arosics imports


def classFactory(iface):
    from .uchange_plugin import UChangePlugin
    return UChangePlugin(iface)
