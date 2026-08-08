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

from .model_registry import (
    MODEL_REGISTRY, is_scd_model, resolve_weights_path,
    SECOND_SEMANTIC_CLASSES, SECOND_SEMANTIC_PALETTE,
)


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

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Binary Change Detection", "Semantic Change Detection"])
        self.mode_selector.currentIndexChanged.connect(self._on_mode_changed)
        model_form.addRow("Detection mode:", self.mode_selector)

        self.device_selector = QComboBox()
        self.device_selector.addItems(["Auto (GPU if available)", "CPU", "GPU"])
        model_form.addRow("Device:", self.device_selector)

        self._model_registry = MODEL_REGISTRY
        self.model_selector = QComboBox()
        self._populate_model_selector()
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
        self.cancel_btn = QPushButton("Close")
        self.cancel_btn.clicked.connect(self.close)
        btn_layout.addStretch()
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

    def _is_scd_mode(self):
        return self.mode_selector.currentIndex() == 1

    def _populate_model_selector(self):
        scd = self._is_scd_mode()
        self.model_selector.clear()
        for name in self._model_registry:
            if is_scd_model(name) == scd:
                self.model_selector.addItem(name)

    def _on_mode_changed(self, _index):
        scd = self._is_scd_mode()
        self._populate_model_selector()
        self.threshold.setVisible(not scd)
        self.min_area.setVisible(not scd)
        self.style_selector.setVisible(not scd)
        proc_form = self.threshold.parent().layout()
        if proc_form:
            for row in range(proc_form.rowCount()):
                label = proc_form.itemAt(row, QFormLayout.LabelRole)
                field = proc_form.itemAt(row, QFormLayout.FieldRole)
                if field and field.widget() in (self.threshold, self.min_area, self.style_selector):
                    if label and label.widget():
                        label.widget().setVisible(not scd)

    def _toggle_custom_weights(self, state):
        custom = bool(state)
        self.weights_row_widget.setVisible(custom)
        self.model_selector.setEnabled(not custom)

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
        if self._is_scd_mode():
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
            label = "output GeoTIFF path" if self._is_scd_mode() else "output GeoPackage path"
            QMessageBox.warning(self, "Error", f"Specify an {label}.")
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
            self._run_inference_subprocess()
        except Exception as e:
            self._log(f"ERROR: {e}\n{traceback.format_exc()}")
            if "out of memory" in str(e).lower():
                QMessageBox.critical(
                    self, "GPU Out of Memory",
                    "The GPU ran out of memory during inference.\n\n"
                    "Try:\n"
                    "- Reducing the tile size (e.g. 128)\n"
                    "- Selecting CPU as the device",
                )
            else:
                self.iface.messageBar().pushCritical("ChangeDetection", f"Inference failed: {e}")
        finally:
            self.run_btn.setEnabled(True)

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
            "--tile-size", str(self.tile_size.value()),
            "--overlap", str(self.overlap.value()),
            "--json-progress",
        ]

        if scd_mode:
            cmd.extend(["--mode", "semantic"])
        else:
            cmd.extend(["--threshold", str(self.threshold.value())])
            if output_path.endswith(".gpkg"):
                cmd.extend(["--output-gpkg", output_path])
                cmd.extend(["--min-area", str(self.min_area.value())])
                style_map = {"Exact": "exact", "Simplified": "simplified", "Convex hull": "convex"}
                cmd.extend(["--style", style_map[self.style_selector.currentText()]])

        if not self.coregister_check.isChecked():
            cmd.append("--no-coreg")

        model_name = self.model_selector.currentText()
        if self._model_registry.get(model_name, {}).get("grayscale", False):
            cmd.append("--grayscale")

        self._log("Starting inference subprocess...")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

        result_info = None
        error_msg = None

        for line in proc.stdout:
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

        if proc.returncode != 0:
            stderr = proc.stderr.read().strip()
            if error_msg:
                raise RuntimeError(error_msg)
            stderr_last = stderr.split("\n")[-1] if stderr else "unknown error"
            raise RuntimeError(f"Inference failed: {stderr_last}")

        if error_msg:
            raise RuntimeError(error_msg)

        if result_info is None:
            raise RuntimeError("Inference produced no result")

        self._set_progress(90)

        if scd_mode:
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

        result_output = result_info.get("output_path", "")
        added_layer = False

        if output_path.endswith(".gpkg") and os.path.isfile(output_path):
            total_polys = result_info.get("total_polys", 0)
            final_polys = result_info.get("final_polys", 0)
            self._log(f"Polygons: {total_polys} created, {final_polys} after filter")

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
                    added_layer = True

        if not added_layer and result_output and os.path.isfile(result_output):
            dest = output_path
            if dest.endswith(".gpkg"):
                dest = os.path.splitext(dest)[0] + os.path.splitext(result_output)[1]
            shutil.copy2(result_output, dest)
            self._log(f"Saved: {dest}")

            if self.add_to_project.isChecked():
                from qgis.core import QgsRasterLayer
                layer = QgsRasterLayer(dest, "Change Detection")
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
                    self._log("Layer added to project.")

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
