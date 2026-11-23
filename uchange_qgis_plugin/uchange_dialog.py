import os
import traceback

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QProgressBar, QTextEdit, QFileDialog,
    QLineEdit, QMessageBox, QGroupBox, QSlider, QLabel,
)
from qgis.PyQt.QtCore import Qt
from qgis.core import (
    QgsApplication, QgsMapLayerProxyModel, QgsProject, QgsVectorLayer,
    QgsSimpleFillSymbolLayer, QgsSymbol, QgsSingleSymbolRenderer,
    QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsUnitTypes,
    QgsCoordinateTransform,
)
from qgis.PyQt.QtGui import QColor
from qgis.gui import QgsExtentGroupBox, QgsMapLayerComboBox

from .model_registry import (
    MODEL_REGISTRY, PRESETS, is_scd_model, is_labelled_model, resolve_weights_path,
    SECOND_SEMANTIC_CLASSES, SECOND_SEMANTIC_PALETTE,
    LANDCOVER_CLASSES, LANDCOVER_PALETTE,
)

# Labelled-layer category for changes with the same land cover on both dates.
_UNCLEAR = "unclear"


def _outline(geometry_type, color, width=0.5, dashed=False):
    """A see-through fill with a coloured outline."""
    symbol = QgsSymbol.defaultSymbol(geometry_type)
    fill = QgsSimpleFillSymbolLayer()
    fill.setColor(QColor(0, 0, 0, 0))
    fill.setStrokeColor(color)
    fill.setStrokeWidth(width)
    if dashed:
        fill.setStrokeStyle(Qt.DashLine)
    symbol.changeSymbolLayer(0, fill)
    return symbol


# filter_score classes: (lower, upper, label, colour, width, dashed)
_SCORE_CLASSES = [
    (0.70, 1.00, "likely real (0.70 and up)", QColor(227, 26, 28), 0.7, False),
    (0.50, 0.70, "probably real (0.50 - 0.70)", QColor(255, 127, 0), 0.6, False),
    (0.00, 0.50, "borderline (under 0.50)", QColor(255, 255, 51), 0.5, True),
]


def _style_changes(layer):
    """Outline changes by the false-alarm filter's score (red, orange, dashed yellow);
    a plain yellow outline when there is no score (filter off)."""
    from qgis.core import QgsGraduatedSymbolRenderer, QgsRendererRange
    gt = layer.geometryType()
    idx = layer.fields().indexOf("filter_score")
    scored = idx >= 0 and any(f[idx] is not None and f[idx] == f[idx] for f in layer.getFeatures())
    if not scored:
        layer.setRenderer(QgsSingleSymbolRenderer(_outline(gt, QColor(255, 255, 0))))
        return
    ranges = [QgsRendererRange(lo, hi, _outline(gt, c, w, d), label)
              for lo, hi, label, c, w, d in _SCORE_CLASSES]          # legend: most sure first
    layer.setRenderer(QgsGraduatedSymbolRenderer("filter_score", ranges))


def _style_labelled(layer):
    """Labelled change: coloured by what each change became; dashed white when the land
    cover is the same on both dates (a change the classes can't name)."""
    categories = []
    for name, (r, g, b) in zip(LANDCOVER_CLASSES[1:], LANDCOVER_PALETTE[1:]):
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        fill = QgsSimpleFillSymbolLayer()
        fill.setColor(QColor(r, g, b, 90))
        fill.setStrokeColor(QColor(r, g, b))
        fill.setStrokeWidth(0.6)
        symbol.changeSymbolLayer(0, fill)
        categories.append(QgsRendererCategory(name, symbol, f"became {name}"))
    # Same class on both dates: a detected change the land-cover classes can't
    # name (e.g. a flat roof read as paving), not "became <class>".
    symbol = QgsSymbol.defaultSymbol(layer.geometryType())
    fill = QgsSimpleFillSymbolLayer()
    fill.setColor(QColor(255, 255, 255, 60))
    fill.setStrokeColor(QColor(255, 255, 255))
    fill.setStrokeWidth(0.6)
    fill.setStrokeStyle(Qt.DashLine)
    symbol.changeSymbolLayer(0, fill)
    categories.append(QgsRendererCategory(_UNCLEAR, symbol, "change, type unclear"))
    layer.setRenderer(QgsCategorizedSymbolRenderer(
        f"CASE WHEN \"from_class\" = \"to_class\" THEN '{_UNCLEAR}' ELSE \"to_class\" END",
        categories))


class _Stopped(Exception):
    """The user stopped the run."""


class UChangeDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent or iface.mainWindow())
        self.iface = iface
        self.setWindowTitle("ChangeDetection")
        self.setMinimumWidth(500)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Input group ---
        input_group = QGroupBox("Input")
        input_form = QFormLayout()

        self.before_layer = QgsMapLayerComboBox()
        self.before_layer.setFilters(QgsMapLayerProxyModel.RasterLayer)
        input_form.addRow("Before raster:", self.before_layer)

        self.after_layer = QgsMapLayerComboBox()
        self.after_layer.setFilters(QgsMapLayerProxyModel.RasterLayer)
        input_form.addRow("After raster:", self.after_layer)

        input_group.setLayout(input_form)
        layout.addWidget(input_group)

        # --- Area to process (off: the whole image) ---
        self.extent_box = QgsExtentGroupBox()
        self.extent_box.setTitle("Process only an area")
        self.extent_box.setCheckable(True)
        self.extent_box.setChecked(False)
        self.extent_box.setToolTip(
            "Run on part of the image only: the current map view, a layer's extent,\n"
            "or a rectangle drawn on the map. Much faster on large images or without\n"
            "a GPU. Outlines can differ slightly from a run on the whole image.")
        self._canvas = self.iface.mapCanvas() if self.iface is not None else None
        self._area_tool = self._prev_tool = self._area_shown = None
        self._drawn = False          # outline only an area drawn on the map, never the default
        if self._canvas is not None:
            # our own drawing tool (orange) replaces the built-in one
            self.extent_box.setMapCanvas(self._canvas, False)
            from .area_tool import area_band
            self._area_shown = area_band(self._canvas)
            self.draw_area_btn = QPushButton("Draw area on map")
            self.draw_area_btn.setToolTip("Drag a rectangle on the map (Esc or right click cancels).")
            self.draw_area_btn.clicked.connect(self._start_drawing)
            box_layout = self.extent_box.layout()
            if box_layout is not None:
                box_layout.addWidget(self.draw_area_btn)
            else:
                layout.addWidget(self.draw_area_btn)
            self.extent_box.extentChanged.connect(self._on_extent_changed)
            self.extent_box.toggled.connect(lambda _on: self._show_area())
        layout.addWidget(self.extent_box)
        self.before_layer.layerChanged.connect(self._on_before_layer_changed)
        self._on_before_layer_changed(self.before_layer.currentLayer())

        # --- Model group ---
        model_group = QGroupBox("Model")
        model_form = QFormLayout()

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Binary Change Detection", "Semantic Change Detection"])
        self.mode_selector.currentIndexChanged.connect(self._on_mode_changed)
        model_form.addRow("Detection mode:", self.mode_selector)

        self.device_selector = QComboBox()
        self.device_selector.addItems(["Auto (GPU if available)", "CPU", "GPU"])
        self.device_selector.currentTextChanged.connect(self._on_device_changed)
        model_form.addRow("Device:", self.device_selector)

        self.max_vram = QSpinBox()
        self.max_vram.setRange(0, 65536)
        self.max_vram.setSingleStep(256)
        self.max_vram.setSuffix(" MB")
        self.max_vram.setSpecialValueText("No limit")
        self.max_vram.setValue(0)
        self.max_vram.setToolTip(
            "Most GPU memory the analysis may use, for small or shared GPUs.\n"
            "Same results, a little slower under a tight limit.\n"
            "The models need at least about 400 MB (plus ~0.3 GB for the\n"
            "GPU driver itself)."
        )
        model_form.addRow("GPU memory limit:", self.max_vram)

        self._model_registry = MODEL_REGISTRY
        self.model_selector = QComboBox()
        self._populate_model_selector()
        self.model_selector.currentTextChanged.connect(self._on_model_changed)
        model_form.addRow("Model:", self.model_selector)

        self.custom_weights_check = QCheckBox("Use custom weights file")
        self.custom_weights_check.stateChanged.connect(self._toggle_custom_weights)
        model_form.addRow("", self.custom_weights_check)

        self.weights_path = QLineEdit()
        self.weights_browse = QPushButton("Browse...")
        self.weights_browse.clicked.connect(self._browse_weights)
        weights_layout = QHBoxLayout()
        weights_layout.addWidget(self.weights_path)
        weights_layout.addWidget(self.weights_browse)
        self.weights_row_widget = QGroupBox()
        self.weights_row_widget.setLayout(weights_layout)
        self.weights_row_widget.setFlat(True)
        self.weights_row_widget.setStyleSheet("QGroupBox { border: none; padding: 0; margin: 0; }")
        self.weights_row_widget.setVisible(False)
        model_form.addRow("Weights:", self.weights_row_widget)

        model_group.setLayout(model_form)
        layout.addWidget(model_group)

        # --- Processing group ---
        proc_group = QGroupBox("Processing")
        proc_form = QFormLayout()

        self.coregister_check = QCheckBox("Co-register images before analysis")
        self.coregister_check.setChecked(True)
        proc_form.addRow("", self.coregister_check)


        self.fast_mode_check = QCheckBox("Fast mode")
        self.fast_mode_check.setChecked(False)
        self.fast_mode_check.setToolTip(
            "About 2x faster, slightly less accurate.\n"
            "Skips the flip averaging the recommended models normally use\n"
            "(each image is processed once instead of four times)."
        )
        proc_form.addRow("", self.fast_mode_check)

        self.filter_check = QCheckBox("Filter false alarms (lean, shadows)")
        self.filter_check.setChecked(True)
        self.filter_check.setToolTip(
            "Finds changes at a low threshold, then a classifier trained on\n"
            "hand-checked changes drops those that look like building lean or\n"
            "shadows; kept changes are outlined at the usual threshold.\n"
            "On hand-checked squares the model had not seen: 69% of changes\n"
            "found with 10 false alarms per km2. Untick to see everything the\n"
            "model finds at its threshold (more changes, many more false alarms)."
        )
        self.filter_check.stateChanged.connect(self._on_filter_toggled)
        self.filter_score = QDoubleSpinBox()
        self.filter_score.setRange(0.05, 0.95)
        self.filter_score.setSingleStep(0.05)
        self.filter_score.setDecimals(2)
        self.filter_score.setValue(0.46)
        self.filter_score.setToolTip(
            "Keep a change if the classifier judges it real with at least this\n"
            "probability. Lower keeps more (more changes, more false alarms).")
        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(self.filter_check)
        filter_layout.addStretch()
        self.filter_score_label = QLabel("keep if real ≥")
        filter_layout.addWidget(self.filter_score_label)
        filter_layout.addWidget(self.filter_score)
        self.filter_widget = QGroupBox()
        self.filter_widget.setLayout(filter_layout)
        self.filter_widget.setFlat(True)
        self.filter_widget.setStyleSheet("QGroupBox { border: none; padding: 0; margin: 0; }")
        proc_form.addRow("", self.filter_widget)
        self._on_filter_toggled(self.filter_check.isChecked())

        self.tile_size = QSpinBox()
        self.tile_size.setRange(128, 1024)
        self.tile_size.setSingleStep(64)
        self.tile_size.setValue(256)
        proc_form.addRow("Tile size:", self.tile_size)

        self.overlap = QSpinBox()
        self.overlap.setRange(0, 256)
        self.overlap.setSingleStep(16)
        self.overlap.setValue(32)
        proc_form.addRow("Tile overlap:", self.overlap)

        self.auto_threshold_check = QCheckBox("Recommended")
        self.auto_threshold_check.setChecked(True)
        self.auto_threshold_check.setToolTip(
            "Use the threshold tested for the selected model on imagery it had\n"
            "not seen. Older models without a tested threshold estimate one per\n"
            "scene instead (less reliable). Untick to set it by hand."
        )
        self.auto_threshold_check.stateChanged.connect(self._toggle_auto_threshold)
        proc_form.addRow("Change threshold:", self.auto_threshold_check)

        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self._on_threshold_slider_changed)
        self.threshold_label = QLabel("0.50")
        self.threshold_label.setMinimumWidth(35)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        self.threshold_widget = QGroupBox()
        self.threshold_widget.setLayout(threshold_layout)
        self.threshold_widget.setFlat(True)
        self.threshold_widget.setStyleSheet("QGroupBox { border: none; padding: 0; margin: 0; }")
        self.threshold_widget.setVisible(False)
        proc_form.addRow("", self.threshold_widget)

        self.min_area = QSpinBox()
        self.min_area.setRange(0, 100000)
        self.min_area.setSingleStep(10)
        self.min_area.setValue(0)
        self.min_area.setSuffix(" m\u00b2")
        self.min_area.setSpecialValueText("No minimum")
        self.min_area.setToolTip("Drop changes smaller than this on the ground (all parts of a change together).")
        proc_form.addRow("Min change area:", self.min_area)

        proc_group.setLayout(proc_form)
        layout.addWidget(proc_group)

        # --- Output group ---
        output_group = QGroupBox("Output")
        output_form = QFormLayout()

        output_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_browse = QPushButton("Browse...")
        self.output_browse.clicked.connect(self._browse_output)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.output_browse)
        self.output_label = "Output GeoPackage:"
        output_form.addRow(self.output_label, output_layout)

        self.add_to_project = QCheckBox("Add result to QGIS project")
        self.add_to_project.setChecked(True)
        output_form.addRow("", self.add_to_project)

        output_group.setLayout(output_form)
        layout.addWidget(output_group)

        # --- Progress ---
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(120)
        layout.addWidget(self.log_output)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop the running analysis.")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_run)
        self.cancel_btn = QPushButton("Close")
        self.cancel_btn.clicked.connect(self.close)
        btn_layout.addStretch()
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.cancel_btn)
        self._proc = None
        self._stopped = False
        layout.addLayout(btn_layout)

        self._on_model_changed(self.model_selector.currentText())

    def _on_before_layer_changed(self, layer):
        """The area is given in the before raster's coordinates; its extent is the default."""
        if layer is None:
            return
        crs = layer.crs()
        was_on = self.extent_box.isChecked()   # setting an extent ticks the box; keep the user's choice
        self._setting_drawn = was_on and self._drawn   # a kept drawn area stays outlined
        try:
            self.extent_box.setOutputCrs(crs)
            self.extent_box.setOriginalExtent(layer.extent(), crs)
            self.extent_box.setCurrentExtent(layer.extent(), crs)
            if not was_on:                     # keep an area the user already chose
                self.extent_box.setOutputExtentFromOriginal()
            self.extent_box.setChecked(was_on)
        finally:
            self._setting_drawn = False

    def _on_extent_changed(self, _rect):
        """Any extent not just drawn (layer, map view, typed) is not outlined."""
        if not self._setting_drawn:
            self._drawn = False
        self._show_area()

    _setting_drawn = False

    def _start_drawing(self):
        from .area_tool import AreaTool, show_rect
        show_rect(self._area_shown, None)      # only the new rectangle while dragging
        self._prev_tool = self._canvas.mapTool()
        self._area_tool = AreaTool(self._canvas)
        self._area_tool.finished.connect(self._area_drawn)
        self._area_tool.cancelled.connect(self._drawing_done)
        self._canvas.setMapTool(self._area_tool)
        self.hide()
        self.iface.messageBar().pushInfo(
            "ChangeDetection", "Drag a rectangle on the map for the area to process (Esc cancels).")

    def _area_drawn(self, rect):
        self._setting_drawn = True
        try:
            self.extent_box.setOutputExtentFromUser(rect, self._canvas.mapSettings().destinationCrs())
            self.extent_box.setChecked(True)
        finally:
            self._setting_drawn = False
        self._drawn = True
        self._drawing_done()

    def _drawing_done(self):
        tool, self._area_tool = self._area_tool, None
        if tool is not None:
            self._canvas.unsetMapTool(tool)
            tool.remove()
        if self._prev_tool is not None:
            self._canvas.setMapTool(self._prev_tool)
        self.iface.messageBar().clearWidgets()
        self.show()
        self.raise_()
        self.activateWindow()
        self._show_area()

    def _show_area(self):
        """Outline the chosen area on the map (orange) while the box is ticked."""
        if self._area_shown is None:
            return
        from .area_tool import show_rect
        rect = None
        if self._drawn and self.extent_box.isChecked() and not self.extent_box.outputExtent().isEmpty():
            ct = QgsCoordinateTransform(self.extent_box.outputCrs(),
                                        self._canvas.mapSettings().destinationCrs(), QgsProject.instance())
            try:
                rect = ct.transformBoundingBox(self.extent_box.outputExtent())
            except Exception:
                rect = None
        show_rect(self._area_shown, rect)

    def closeEvent(self, event):
        self._stop_run()                       # closing the dialog ends a running analysis
        if self._area_tool is not None:
            self._drawing_done()
        if self._area_shown is not None:
            self._canvas.scene().removeItem(self._area_shown)
            self._area_shown = None
        super().closeEvent(event)

    def _area_problem(self):
        """Why the chosen area can't be processed, or None."""
        if not self.extent_box.isChecked():
            return None
        layer = self.before_layer.currentLayer()
        try:
            from osgeo import gdal
            ds = gdal.Open(layer.source())
            georef = ds is not None and ds.GetGeoTransform(can_return_null=True) is not None
        except Exception:
            georef = True                      # can't tell: let the run report it
        if not georef or not layer.crs().isValid():
            return ("\"Process only an area\" needs georeferenced rasters (such as GeoTIFFs with a\n"
                    "coordinate system); the before raster has no georeferencing.\n"
                    "Untick the box to process the whole image.")
        r = self.extent_box.outputExtent()
        if r.isEmpty() or not r.intersects(layer.extent()):
            return "The area to process does not overlap the before raster."
        return None

    def _area_arg(self):
        """--extent value for the chosen area, or None for the whole image."""
        if not self.extent_box.isChecked():
            return None
        r = self.extent_box.outputExtent()
        if r.isEmpty():
            return None
        return f"{r.xMinimum()!r},{r.yMinimum()!r},{r.xMaximum()!r},{r.yMaximum()!r}"

    def _area_km2(self):
        """Size of what will be processed in km2 (chosen area or the before raster), or None."""
        layer = self.before_layer.currentLayer()
        if layer is None or layer.crs().isGeographic():
            return None
        r = self.extent_box.outputExtent() if self.extent_box.isChecked() else layer.extent()
        r = r.intersect(layer.extent())
        to_m = QgsUnitTypes.fromUnitToUnitFactor(layer.crs().mapUnits(), QgsUnitTypes.DistanceMeters)
        return r.width() * r.height() * to_m * to_m / 1e6 if not r.isEmpty() else None

    def _is_scd_mode(self):
        return self.mode_selector.currentIndex() == 1

    def _is_labelled(self):
        """Semantic mode with a model that writes labelled polygons."""
        return (self._is_scd_mode() and not self.custom_weights_check.isChecked()
                and is_labelled_model(self.model_selector.currentText()))

    def _raster_output(self):
        return self._is_scd_mode() and not self._is_labelled()

    def _populate_model_selector(self):
        scd = self._is_scd_mode()
        self.model_selector.clear()
        for name in self._model_registry:
            if is_scd_model(name) == scd:
                self.model_selector.addItem(name)

    def _toggle_auto_threshold(self, state):
        self.threshold_widget.setVisible(not bool(state))

    def _on_threshold_slider_changed(self, value):
        self.threshold_label.setText(f"{value / 100:.2f}")

    def _on_mode_changed(self, _index):
        self._populate_model_selector()
        self._on_mode_changed_visibility()

    def _has_preset(self):
        """The selected model runs with its tested settings (tiles, scale, threshold)."""
        entry = self._model_registry.get(self.model_selector.currentText(), {})
        return bool(entry.get("preset")) and not self.custom_weights_check.isChecked()

    def _uses_preset(self):
        """The selected model runs with a preset that offers Fast mode."""
        entry = self._model_registry.get(self.model_selector.currentText(), {})
        preset = PRESETS.get(entry.get("preset"), {})
        return bool(preset.get("fast_mode")) and not self.custom_weights_check.isChecked()

    def _uses_filter(self):
        """The selected model's preset offers the false-alarm filter, and its weights exist."""
        entry = self._model_registry.get(self.model_selector.currentText(), {})
        bf = PRESETS.get(entry.get("preset"), {}).get("false_alarm_filter")
        if not bf or self.custom_weights_check.isChecked():
            return False
        root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        return os.path.isfile(os.path.join(root, bf["file"]))

    def _on_filter_toggled(self, state):
        on = bool(state)
        self.filter_score.setVisible(on)
        self.filter_score_label.setVisible(on)

    def _on_mode_changed_visibility(self):
        self.fast_mode_check.setVisible(self._uses_preset())
        # Models with a preset use the tile size and overlap they were tested with.
        tiles = not self._has_preset()
        for w in (self.tile_size, self.overlap):
            w.setVisible(tiles)
            lab = w.parent().layout().labelForField(w) if w.parent() and w.parent().layout() else None
            if lab is not None:
                lab.setVisible(tiles)
        self.filter_widget.setVisible(self._uses_filter() and not self._raster_output())
        # Threshold and min-area apply to everything that writes polygons.
        scd = self._raster_output()
        self.auto_threshold_check.setVisible(not scd)
        self.threshold_widget.setVisible(not scd and not self.auto_threshold_check.isChecked())
        self.min_area.setVisible(not scd)
        proc_form = self.auto_threshold_check.parent().layout()
        if proc_form:
            for row in range(proc_form.rowCount()):
                label = proc_form.itemAt(row, QFormLayout.LabelRole)
                field = proc_form.itemAt(row, QFormLayout.FieldRole)
                if field and field.widget() in (self.auto_threshold_check, self.threshold_widget, self.min_area):
                    if label and label.widget():
                        label.widget().setVisible(not scd)

    def _on_model_changed(self, name):
        if name:
            self._on_mode_changed_visibility()
        entry = self._model_registry.get(name, {})
        preferred_tile = entry.get("tile_size")
        if preferred_tile:
            self.tile_size.setValue(preferred_tile)
        preferred_overlap = entry.get("overlap")
        if preferred_overlap is not None:
            self.overlap.setValue(preferred_overlap)

    def _toggle_custom_weights(self, state):
        custom = bool(state)
        self.weights_row_widget.setVisible(custom)
        self.model_selector.setEnabled(not custom)
        self._on_mode_changed_visibility()

    def _get_weights_path(self):
        if self.custom_weights_check.isChecked() and self.weights_path.text():
            return self.weights_path.text()
        return resolve_weights_path(self.model_selector.currentText())

    def _browse_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model weights", "", "PyTorch weights (*.pth)"
        )
        if path:
            self.weights_path.setText(path)

    def _browse_output(self):
        if self._raster_output():
            path, _ = QFileDialog.getSaveFileName(
                self, "Save SCD output", "", "GeoTIFF (*.tif)"
            )
            if path:
                if not path.endswith(".tif"):
                    path += ".tif"
                self.output_path.setText(path)
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save GeoPackage", "", "GeoPackage (*.gpkg)"
            )
            if path:
                if not path.endswith(".gpkg"):
                    path += ".gpkg"
                self.output_path.setText(path)

    def _log(self, msg):
        self.log_output.append(msg)
        QgsApplication.processEvents()

    def _set_progress(self, val):
        self.progress.setValue(val)
        QgsApplication.processEvents()

    def _validate(self):
        before = self.before_layer.currentLayer()
        after = self.after_layer.currentLayer()
        if not before or not after:
            QMessageBox.warning(self, "Error", "Select both before and after raster layers.")
            return False
        weights = self._get_weights_path()
        if not weights or not os.path.isfile(weights):
            QMessageBox.warning(self, "Error",
                f"Model weights not found:\n{weights}\n\n"
                "Run the installer to download weights, or use a custom weights file.")
            return False
        if not self.output_path.text():
            label = "output GeoTIFF path" if self._raster_output() else "output GeoPackage path"
            QMessageBox.warning(self, "Error", f"Specify an {label}.")
            return False
        problem = self._area_problem()
        if problem:
            QMessageBox.warning(self, "Area to process", problem)
            return False
        return True

    def _run_coregistration_subprocess(self, ref_path, tgt_path):
        import subprocess
        import json
        try:
            from ._env_config import VENV_PYTHON
        except ImportError:
            return None

        if not os.path.isfile(VENV_PYTHON):
            return None

        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = (
            "import sys, json, os\n"
            f"sys.path.insert(0, {repr(project_dir)})\n"
            "from uchange_qgis_plugin.coregistration import coregister_images\n"
            f"r = coregister_images({repr(ref_path)}, {repr(tgt_path)}, "
            "max_shift=50, window_size=(1024, 1024))\n"
            "json.dump({'success': r.success, 'corrected_path': r.corrected_path,\n"
            "  'shift_x_px': r.shift_x_px, 'shift_y_px': r.shift_y_px,\n"
            "  'message': r.message}, sys.stdout)\n"
        )
        try:
            proc = subprocess.run(
                [VENV_PYTHON, "-c", script],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            return {"error": "Co-registration timed out (120s)"}

        if proc.returncode != 0:
            if proc.returncode < 0:
                return {"error": "Co-registration ran out of memory (images may be too large)"}
            stderr = proc.stderr.strip().split("\n")[-1] if proc.stderr else "unknown error"
            return {"error": f"Co-registration failed: {stderr}"}

        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {"error": "Co-registration returned invalid output"}

        if not result.get("success"):
            return {"error": result.get("message", "Co-registration failed")}

        return result

    def _on_device_changed(self, text):
        # The memory limit only means something on a GPU.
        on_gpu = text != "CPU"
        self.max_vram.setVisible(on_gpu)
        label = self.max_vram.parent().layout().labelForField(self.max_vram)
        if label:
            label.setVisible(on_gpu)

    def _resolve_device_str(self):
        device_text = self.device_selector.currentText()
        if device_text == "GPU":
            return "gpu"
        elif device_text == "CPU":
            return "cpu"
        else:
            return "auto"

    def _on_run(self):
        if not self._validate():
            return

        device_text = self.device_selector.currentText()
        if device_text == "CPU":
            km2 = self._area_km2()
            size = (f"For this area ({km2:.2f} km\u00b2) expect roughly {max(1, round(30 * km2))} minutes"
                    f"{'' if not self._uses_preset() else f', or about {max(1, round(18 * km2))} with Fast mode'}"
                    ".\n" if km2 else "")
            reply = QMessageBox.question(
                self, "Running on the CPU",
                "On a CPU this takes about half an hour per km\u00b2 (measured on a 24-thread\n"
                "processor); a GPU needs under a minute.\n" + size +
                "\"Process only an area\" lets you try a small part first.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.log_output.clear()
        QgsApplication.processEvents()

        try:
            self._run_inference_subprocess()
        except _Stopped:
            self._log("Stopped.")
            self.progress.setValue(0)
        except Exception as e:
            self._log(f"ERROR: {e}\n{traceback.format_exc()}")
            if "out of memory" in str(e).lower():
                tips = ["- Closing other programs that use the GPU",
                        "- Selecting CPU as the device"]
                if self.max_vram.value():
                    tips.insert(0, "- Raising the GPU memory limit (at least about 400 MB)")
                QMessageBox.critical(
                    self, "GPU Out of Memory",
                    "The GPU ran out of memory during inference.\n\n"
                    "Try:\n" + "\n".join(tips),
                )
            else:
                self.iface.messageBar().pushCritical("ChangeDetection", f"Inference failed: {e}")
        finally:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.kill()              # a failed run leaves nothing running
            self._proc = None
            self.stop_btn.setEnabled(False)
            self.run_btn.setEnabled(True)

    def _stop_run(self):
        """End the running analysis (its subprocess); the run loop then reports it."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self._stopped = True
        self.stop_btn.setEnabled(False)
        self._log("Stopping...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    def _run_inference_subprocess(self):
        import subprocess
        import json
        import tempfile

        try:
            from ._env_config import VENV_PYTHON
        except ImportError:
            raise RuntimeError("Plugin not installed correctly: _env_config.py missing. Re-run install.sh.")

        if not os.path.isfile(VENV_PYTHON):
            raise RuntimeError(f"Venv Python not found: {VENV_PYTHON}\nRe-run install.sh.")

        project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        script_path = os.path.join(project_dir, "detect_changes.py")
        if not os.path.isfile(script_path):
            raise RuntimeError(f"detect_changes.py not found: {script_path}")

        before_path = self.before_layer.currentLayer().source()
        after_path = self.after_layer.currentLayer().source()
        output_path = self.output_path.text()
        weights = self._get_weights_path()
        device = self._resolve_device_str()
        scd_mode = self._is_scd_mode()

        output_dir = tempfile.mkdtemp(prefix="uchange_")

        cmd = [
            VENV_PYTHON, script_path,
            "--before", before_path,
            "--after", after_path,
            "--output", output_dir,
            "--weights", weights,
            "--device", device,
            "--json-progress",
        ]
        if not self._has_preset():          # a preset brings the tile settings it was tested with
            cmd.extend(["--tile-size", str(self.tile_size.value()), "--overlap", str(self.overlap.value())])
        if self.max_vram.value() and device != "cpu":
            cmd.extend(["--max-vram", str(self.max_vram.value())])
        area = self._area_arg()
        if area:
            cmd.extend(["--extent", area])
            self._log(f"Area to process: {area} ({self.extent_box.outputCrs().authid() or 'raster coordinates'})")

        if scd_mode and not self._is_labelled():
            cmd.extend(["--mode", "semantic"])
        else:
            if self.auto_threshold_check.isChecked():
                cmd.extend(["--threshold", "auto"])
            else:
                cmd.extend(["--threshold", f"{self.threshold_slider.value() / 100:.2f}"])
            gpkg_path = output_path
            if not gpkg_path.endswith(".gpkg"):
                gpkg_path = os.path.splitext(gpkg_path)[0] + ".gpkg"
            cmd.extend(["--output-gpkg", gpkg_path])
            if self.min_area.value():
                cmd.extend(["--min-area-m2", str(self.min_area.value())])

        if not self.coregister_check.isChecked():
            cmd.append("--no-coreg")


        model_name = self.model_selector.currentText()
        model_entry = self._model_registry.get(model_name, {})
        if model_entry.get("grayscale", False):
            cmd.append("--grayscale")
        if model_entry.get("type"):
            cmd.extend(["--model-type", model_entry["type"]])
        if model_entry.get("preset") and not self.custom_weights_check.isChecked():
            cmd.extend(["--preset", model_entry["preset"]])
            if self.fast_mode_check.isChecked():
                cmd.append("--fast")
            if self._uses_filter() and not self._raster_output():
                if self.filter_check.isChecked():
                    cmd.extend(["--filter-false-alarms",
                                "--filter-min-score", f"{self.filter_score.value():.2f}"])
                else:
                    cmd.append("--no-filter-false-alarms")

        self._log("Starting inference subprocess...")
        import queue
        import threading
        # stderr goes to a file: a pipe read only at the end could fill up and hang the run
        err_file = tempfile.TemporaryFile(mode="w+")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=err_file,
            text=True, bufsize=1,
        )
        self._proc, self._stopped = proc, False
        self.stop_btn.setEnabled(True)

        # stdout is read in a thread so the dialog (and Stop) stays responsive
        # even while the analysis prints nothing for a while
        lines = queue.Queue()

        def _reader():
            for ln in proc.stdout:
                lines.put(ln)
            lines.put(None)

        threading.Thread(target=_reader, daemon=True).start()

        result_info = None
        error_msg = None

        while True:
            try:
                line = lines.get(timeout=0.05)
            except queue.Empty:
                QgsApplication.processEvents()
                continue
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._log(line)
                continue

            msg_type = msg.get("type")
            if msg_type == "progress":
                self._set_progress(msg.get("percent", 0))
            elif msg_type == "log":
                self._log(msg.get("message", ""))
            elif msg_type == "device":
                self._log(f"Device: {msg.get('device', '?')}")
                if device == "gpu" and msg.get("device") != "cuda":
                    proc.terminate()
                    raise RuntimeError(
                        "CUDA GPU is not available on this system.\n"
                        "Select 'Auto' or 'CPU' instead.")
            elif msg_type == "error":
                error_msg = msg.get("message", "Unknown error")
            elif msg_type == "result":
                result_info = msg

        proc.wait()
        self._proc = None
        self.stop_btn.setEnabled(False)
        err_file.seek(0)
        stderr = err_file.read().strip()
        err_file.close()
        if self._stopped:
            raise _Stopped()

        if proc.returncode != 0:
            if error_msg:
                raise RuntimeError(error_msg)
            stderr_last = stderr.split("\n")[-1] if stderr else "unknown error"
            raise RuntimeError(f"Inference failed: {stderr_last}")

        if error_msg:
            raise RuntimeError(error_msg)

        if result_info is None:
            raise RuntimeError("Inference produced no result")

        self._set_progress(90)

        if result_info.get("mode") == "labelled":
            self._handle_labelled_result(result_info, output_path, output_dir)
        elif scd_mode:
            self._handle_scd_result(result_info, output_dir)
        else:
            self._handle_binary_result(result_info, output_path, output_dir)

        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)

        self._set_progress(100)
        mode_label = "Semantic change" if scd_mode else "Change"
        self._log("Done!")
        self.iface.messageBar().pushSuccess("ChangeDetection", f"{mode_label} detection complete!")

    def _handle_binary_result(self, result_info, output_path, output_dir):
        import shutil

        threshold = result_info.get("threshold")
        if threshold is not None and self.auto_threshold_check.isChecked():
            self._log(f"Threshold used: {threshold:.6f}")

        gpkg_path = output_path
        if not gpkg_path.endswith(".gpkg"):
            gpkg_path = os.path.splitext(gpkg_path)[0] + ".gpkg"

        if os.path.isfile(gpkg_path):
            total_polys = result_info.get("total_polys", 0)
            final_polys = result_info.get("final_polys", 0)
            self._log(f"Changes written: {final_polys} (of {total_polys}; the rest under the minimum area)")

            if self.add_to_project.isChecked():
                layer = QgsVectorLayer(gpkg_path, "Change Detection", "ogr")
                if layer.isValid():
                    _style_changes(layer)
                    QgsProject.instance().addMapLayer(layer)
                    self._log("Layer added to project.")
        else:
            result_output = result_info.get("output_path", "")
            if result_output and os.path.isfile(result_output):
                dest = output_path
                shutil.copy2(result_output, dest)
                self._log(f"Saved: {dest}")

                if self.add_to_project.isChecked():
                    from qgis.core import QgsRasterLayer
                    layer = QgsRasterLayer(dest, "Change Detection")
                    if layer.isValid():
                        QgsProject.instance().addMapLayer(layer)
                        self._log("Layer added to project.")

    def _handle_labelled_result(self, result_info, output_path, output_dir):
        """Polygons coloured by what each change became; before/after rasters beside them."""
        import shutil

        gpkg_path = result_info.get("output_path") or output_path
        base = os.path.splitext(gpkg_path)[0]
        for tag in ("from", "to"):
            src = result_info.get(f"{tag}_path")
            if src and os.path.isfile(src):
                dest = f"{base}_landcover_{tag}.tif"
                shutil.copy2(src, dest)
                self._log(f"Saved: {dest}")
        if not os.path.isfile(gpkg_path):
            return
        self._log(f"Changes written: {result_info.get('final_polys', 0)} (of "
                  f"{result_info.get('total_polys', 0)}; the rest under the minimum area)")
        if not self.add_to_project.isChecked():
            return
        layer = QgsVectorLayer(gpkg_path, "Labelled change", "ogr")
        if not layer.isValid():
            return
        _style_labelled(layer)
        QgsProject.instance().addMapLayer(layer)
        self._log("Layer 'Labelled change' added (colour = what it became; dashed white = "
                  "change whose type is unclear; see the 'change' field for from -> to).")

    def _handle_scd_result(self, result_info, output_dir):
        output_base = self.output_path.text()
        dest_dir = os.path.dirname(output_base) or "."
        os.makedirs(dest_dir, exist_ok=True)

        import shutil

        binary_path = os.path.join(output_dir, "binary_change.tif")
        semantic_path = os.path.join(output_dir, "semantic_change.tif")
        dest_binary = os.path.join(dest_dir, "binary_change.tif")
        dest_semantic = os.path.join(dest_dir, "semantic_change.tif")

        if os.path.isfile(binary_path):
            shutil.copy2(binary_path, dest_binary)
            self._log(f"Saved: {dest_binary}")
        if os.path.isfile(semantic_path):
            shutil.copy2(semantic_path, dest_semantic)
            self._log(f"Saved: {dest_semantic}")

        if self.add_to_project.isChecked():
            from qgis.core import QgsRasterLayer, QgsPalettedRasterRenderer

            if os.path.isfile(dest_binary):
                mask_layer = QgsRasterLayer(dest_binary, "Binary Change")
                if mask_layer.isValid():
                    QgsProject.instance().addMapLayer(mask_layer)
                    self._log("Layer 'Binary Change' added.")

            if os.path.isfile(dest_semantic):
                sem_layer = QgsRasterLayer(dest_semantic, "Semantic Change")
                if sem_layer.isValid():
                    classes = []
                    for i in range(1, len(SECOND_SEMANTIC_CLASSES)):
                        r, g, b = SECOND_SEMANTIC_PALETTE[i]
                        classes.append(QgsPalettedRasterRenderer.Class(
                            i, QColor(r, g, b), SECOND_SEMANTIC_CLASSES[i]))
                    renderer = QgsPalettedRasterRenderer(
                        sem_layer.dataProvider(), 1, classes)
                    sem_layer.setRenderer(renderer)
                    QgsProject.instance().addMapLayer(sem_layer)
                    self._log("Layer 'Semantic Change' added.")
