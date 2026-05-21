def classFactory(iface):
    from .uchange_plugin import UChangePlugin
    return UChangePlugin(iface)
