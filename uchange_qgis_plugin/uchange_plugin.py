from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon

import os


class UChangePlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dialog = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        self.action = QAction(icon, "ChangeDetection", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu("&ChangeDetection", self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        self.iface.removePluginMenu("&ChangeDetection", self.action)
        self.iface.removeToolBarIcon(self.action)

    def run(self):
        from .uchange_dialog import UChangeDialog
        self.dialog = UChangeDialog(self.iface)
        self.dialog.show()
