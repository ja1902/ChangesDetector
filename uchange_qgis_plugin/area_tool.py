"""Drawing the area to process on the map: an orange, see-through rectangle."""
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.core import QgsPointXY, QgsRectangle, QgsWkbTypes
from qgis.gui import QgsMapTool, QgsRubberBand

FILL = QColor(255, 140, 0, 80)      # orange, about a third opaque
EDGE = QColor(255, 110, 0, 255)


def area_band(canvas):
    """A rubber band styled for the area: orange fill, solid orange edge."""
    band = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
    band.setFillColor(FILL)
    band.setStrokeColor(EDGE)
    band.setWidth(2)
    return band


def show_rect(band, rect):
    """Draw rect (map canvas coordinates) on band, or clear it for None."""
    band.reset(QgsWkbTypes.PolygonGeometry)
    if rect is None or rect.isEmpty():
        return
    for x, y in ((rect.xMinimum(), rect.yMinimum()), (rect.xMaximum(), rect.yMinimum()),
                 (rect.xMaximum(), rect.yMaximum()), (rect.xMinimum(), rect.yMaximum())):
        band.addPoint(QgsPointXY(x, y), False)
    band.closePoints(True)


class AreaTool(QgsMapTool):
    """Press, drag and release to draw a rectangle; Esc or right click cancels.

    finished(QgsRectangle) in map canvas coordinates, or cancelled().
    """
    finished = pyqtSignal(QgsRectangle)
    cancelled = pyqtSignal()

    def __init__(self, canvas):
        super().__init__(canvas)
        self.band = area_band(canvas)
        self.start = None
        self.setCursor(Qt.CrossCursor)

    def canvasPressEvent(self, e):
        if e.button() == Qt.RightButton:
            self._cancel()
            return
        self.start = self.toMapCoordinates(e.pos())
        show_rect(self.band, None)

    def canvasMoveEvent(self, e):
        if self.start is not None:
            show_rect(self.band, QgsRectangle(self.start, self.toMapCoordinates(e.pos())))

    def canvasReleaseEvent(self, e):
        if self.start is None or e.button() != Qt.LeftButton:
            return
        rect = QgsRectangle(self.start, self.toMapCoordinates(e.pos()))
        self.start = None
        show_rect(self.band, None)
        if rect.width() > 0 and rect.height() > 0:
            self.finished.emit(rect)
        else:
            self.cancelled.emit()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self._cancel()

    def _cancel(self):
        self.start = None
        show_rect(self.band, None)
        self.cancelled.emit()

    def deactivate(self):
        show_rect(self.band, None)
        super().deactivate()

    def remove(self):
        self.canvas().scene().removeItem(self.band)
