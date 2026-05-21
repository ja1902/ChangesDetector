import os
import traceback

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QProgressBar, QTextEdit, QFileDialog,
    QLineEdit, QMessageBox, QGroupBox,
)
from qgis.core import (
    QgsApplication, QgsMapLayerProxyModel, QgsProject, QgsVectorLayer,
    QgsSimpleFillSymbolLayer, QgsSymbol, QgsSingleSymbolRenderer,
)
from qgis.PyQt.QtGui import QColor
from qgis.gui import QgsMapLayerComboBox


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

        # --- Model group ---
        model_group = QGroupBox("Model")
        model_form = QFormLayout()

        self.device_selector = QComboBox()
        self.device_selector.addItems(["Auto (GPU if available)", "CPU", "GPU"])
        model_form.addRow("Device:", self.device_selector)

        from .model_bridge import MODEL_REGISTRY
        self.model_selector = QComboBox()
        self.model_selector.addItems(list(MODEL_REGISTRY.keys()))
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

        self.tile_size = QSpinBox()
        self.tile_size.setRange(128, 1024)
        self.tile_size.setSingleStep(64)
        self.tile_size.setValue(256)
        proc_form.addRow("Tile size:", self.tile_size)

        self.overlap = QSpinBox()
        self.overlap.setRange(0, 256)
        self.overlap.setSingleStep(16)
        self.overlap.setValue(0)
        proc_form.addRow("Tile overlap:", self.overlap)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.5)
        self.threshold.setDecimals(2)
        proc_form.addRow("Change threshold:", self.threshold)

        self.min_area = QSpinBox()
        self.min_area.setRange(0, 10000)
        self.min_area.setValue(0)
        self.min_area.setSuffix(" px")
        proc_form.addRow("Min polygon area:", self.min_area)

        self.style_selector = QComboBox()
        self.style_selector.addItems(["Exact", "Simplified", "Convex hull"])
        proc_form.addRow("Polygon style:", self.style_selector)

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
        output_form.addRow("Output GeoPackage:", output_layout)

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
        self.cancel_btn = QPushButton("Close")
        self.cancel_btn.clicked.connect(self.close)
        btn_layout.addStretch()
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

    def _toggle_custom_weights(self, state):
        custom = bool(state)
        self.weights_row_widget.setVisible(custom)
        self.model_selector.setEnabled(not custom)

    def _get_weights_path(self):
        if self.custom_weights_check.isChecked() and self.weights_path.text():
            return self.weights_path.text()
        from .model_bridge import resolve_weights_path
        return resolve_weights_path(self.model_selector.currentText())

    def _browse_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model weights", "", "PyTorch weights (*.pth)"
        )
        if path:
            self.weights_path.setText(path)

    def _browse_output(self):
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
            QMessageBox.warning(self, "Error", "Specify an output GeoPackage path.")
            return False
        return True

    def _resolve_device(self):
        from .model_bridge import _ensure_venv_on_path
        _ensure_venv_on_path()
        import torch

        selection = self.device_selector.currentText()
        if selection == "CPU":
            return torch.device("cpu")
        elif selection == "GPU":
            if not torch.cuda.is_available():
                QMessageBox.warning(self, "Error", "CUDA GPU is not available on this system.")
                return None
            return torch.device("cuda")
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _on_run(self):
        if not self._validate():
            return

        device = self._resolve_device()
        if device is None:
            return

        if device.type == "cpu":
            reply = QMessageBox.question(
                self, "Performance Warning",
                "CPU inference may be slow.\n"
                "Consider using a GPU if available.\n\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.log_output.clear()
        QgsApplication.processEvents()

        try:
            self._run_inference(device)
        except Exception as e:
            self._log(f"ERROR: {e}\n{traceback.format_exc()}")
            self.iface.messageBar().pushCritical("ChangeDetection", f"Inference failed: {e}")
        finally:
            self.run_btn.setEnabled(True)

    def _run_inference(self, device):
        import numpy as np
        from .model_bridge import _ensure_venv_on_path
        _ensure_venv_on_path()
        import torch
        from .model_bridge import build_model
        from .raster_io import read_raster, polygonize_mask
        from .tiling import run_tiled_inference

        before_layer = self.before_layer.currentLayer()
        after_layer = self.after_layer.currentLayer()

        self._log("Reading before raster...")
        pre_img, geotransform, projection_wkt = read_raster(before_layer.source())
        if not projection_wkt:
            projection_wkt = before_layer.crs().toWkt()
            extent = before_layer.extent()
            pixel_w = extent.width() / before_layer.width()
            pixel_h = extent.height() / before_layer.height()
            geotransform = (
                extent.xMinimum(), pixel_w, 0.0,
                extent.yMaximum(), 0.0, -pixel_h,
            )
        self._set_progress(5)

        self._log("Reading after raster...")
        post_img, _, _ = read_raster(after_layer.source())
        self._set_progress(10)

        if pre_img.shape[:2] != post_img.shape[:2]:
            raise ValueError(
                f"Image dimensions don't match: "
                f"before={pre_img.shape[:2]}, after={post_img.shape[:2]}"
            )

        self._log(f"Using device: {device}")
        self._log("Building MambaBCD model...")
        model, load_summary = build_model(self._get_weights_path(), device)
        self._log(load_summary)
        self._set_progress(20)

        h, w = pre_img.shape[:2]
        tile_size = self.tile_size.value()
        overlap = self.overlap.value()
        self._log(f"Running tiled inference ({w}x{h}, tile={tile_size}, overlap={overlap})...")

        def progress_fn(current, total):
            pct = 20 + int(60 * current / total)
            self._set_progress(pct)

        prob_map = run_tiled_inference(
            model, pre_img, post_img,
            tile_size=tile_size,
            overlap=overlap,
            device=device,
            progress_fn=progress_fn,
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        change_pixels = int((prob_map > 0.5).sum())
        total_pixels = prob_map.size
        self._log(
            f"Prob map stats: min={prob_map.min():.4f}, max={prob_map.max():.4f}, "
            f"mean={prob_map.mean():.4f}"
        )
        self._log(f"Pixels > 0.5: {change_pixels}/{total_pixels} ({change_pixels/total_pixels:.2%})")

        threshold = self.threshold.value()
        binary_mask = (prob_map > threshold).astype(np.uint8)

        n_change = int(binary_mask.sum())
        self._log(f"Binary mask at threshold {threshold}: {n_change} change pixels")
        self._set_progress(85)

        output_path = self.output_path.text()
        self._log(f"Polygonizing to {output_path}...")
        min_area = self.min_area.value()
        pixel_w = abs(geotransform[1])
        pixel_h = abs(geotransform[5])
        min_area_map = min_area * pixel_w * pixel_h

        style_map = {"Exact": "exact", "Simplified": "simplified", "Convex hull": "convex hull"}
        style = style_map[self.style_selector.currentText()]

        total_polys, final_polys = polygonize_mask(
            binary_mask, geotransform, projection_wkt,
            output_path, min_area=min_area_map, style=style,
        )
        self._log(f"Polygons: {total_polys} created, {final_polys} after min-area filter")
        self._set_progress(100)
        self._log("Done!")

        if self.add_to_project.isChecked():
            layer = QgsVectorLayer(output_path, "Change Detection", "ogr")
            if layer.isValid():
                symbol = QgsSymbol.defaultSymbol(layer.geometryType())
                fill = QgsSimpleFillSymbolLayer()
                fill.setColor(QColor(0, 0, 0, 0))
                fill.setStrokeColor(QColor(255, 255, 0))
                fill.setStrokeWidth(0.5)
                symbol.changeSymbolLayer(0, fill)
                layer.setRenderer(QgsSingleSymbolRenderer(symbol))
                QgsProject.instance().addMapLayer(layer)
                self._log("Layer added to project.")
            else:
                self._log("Warning: could not load output layer.")

        self.iface.messageBar().pushSuccess("ChangeDetection", "Change detection complete!")
