"""ChangeDetection in the QGIS Processing toolbox: batch runs over many pairs and use in
graphical models. Runs detect_changes.py exactly as the dialog does."""
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm, QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface, QgsProcessingParameterBoolean,
    QgsProcessingParameterDefinition, QgsProcessingParameterEnum, QgsProcessingParameterExtent,
    QgsProcessingParameterNumber, QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorDestination, QgsProcessingProvider,
)
from qgis.PyQt.QtGui import QIcon

from .model_registry import MODEL_REGISTRY, PRESETS, resolve_weights_path

_HERE = os.path.dirname(__file__)
# binary models first, then labelled (from -> to); the recommended one is the default
MODELS = sorted(MODEL_REGISTRY, key=lambda n: (MODEL_REGISTRY[n]["type"] == "dinov2_lc",
                                               "recommended" not in n))
DEVICES = ["Auto (GPU if available)", "CPU", "GPU"]


class _Styler(QgsProcessingLayerPostProcessorInterface):
    """Gives the loaded result the dialog's styling."""

    def __init__(self, labelled):
        super().__init__()
        self.labelled = labelled

    def postProcessLayer(self, layer, context, feedback):
        from .uchange_dialog import _style_changes, _style_labelled
        (_style_labelled if self.labelled else _style_changes)(layer)


# Post-processors must outlive the algorithm instance (Processing does not take
# ownership and runs them after the instance is gone): one each, kept here.
_STYLERS = {}


def _styler(labelled):
    if labelled not in _STYLERS:
        _STYLERS[labelled] = _Styler(labelled)
    return _STYLERS[labelled]


class DetectChangesAlgorithm(QgsProcessingAlgorithm):
    BEFORE, AFTER, MODEL, EXTENT = "BEFORE", "AFTER", "MODEL", "EXTENT"
    FILTER, MIN_SCORE, FAST, MIN_AREA, DEVICE, OUTPUT = (
        "FILTER", "MIN_SCORE", "FAST", "MIN_AREA", "DEVICE", "OUTPUT")

    def name(self):
        return "detectchanges"

    def displayName(self):
        return "Detect changes"

    def createInstance(self):
        return DetectChangesAlgorithm()

    def shortHelpString(self):
        return (
            "Finds what changed between two georeferenced images of the same place and writes "
            "one polygon per change to a GeoPackage, with a confidence and, with the "
            "false-alarm filter, a filter_score (probability the change is real).<br><br>"
            "The labelled model also writes from_class / to_class (what it changed from and into) "
            "and saves before/after land-cover rasters next to the output.<br><br>"
            "Use <b>Run as Batch Process</b> to process many pairs. On a CPU expect about "
            "half an hour per km²; set an area to process only part of the image.")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.BEFORE, "Before raster"))
        self.addParameter(QgsProcessingParameterRasterLayer(self.AFTER, "After raster"))
        self.addParameter(QgsProcessingParameterEnum(self.MODEL, "Model", options=MODELS, defaultValue=0))
        self.addParameter(QgsProcessingParameterExtent(
            self.EXTENT, "Process only this area (empty: the whole image)", optional=True))
        self.addParameter(QgsProcessingParameterBoolean(
            self.FILTER, "Filter false alarms (lean, shadows)", defaultValue=True))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_SCORE, "Keep a change if judged real with at least",
            type=QgsProcessingParameterNumber.Double, defaultValue=0.46, minValue=0.05, maxValue=0.95))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_AREA, "Minimum change area (m², 0: none)",
            type=QgsProcessingParameterNumber.Double, defaultValue=0.0, minValue=0.0))
        self.addParameter(QgsProcessingParameterBoolean(
            self.FAST, "Fast mode (about 2x faster on the model step, slightly less accurate)",
            defaultValue=False))
        device = QgsProcessingParameterEnum(self.DEVICE, "Device", options=DEVICES, defaultValue=0)
        device.setFlags(device.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(device)
        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT, "Changes", type=QgsProcessing.TypeVectorPolygon))

    def _command(self, parameters, context):
        try:
            from ._env_config import VENV_PYTHON
        except ImportError:
            raise QgsProcessingException("Plugin not installed correctly (_env_config.py missing): "
                                         "re-run the installer.")
        script = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                              "detect_changes.py")
        before = self.parameterAsRasterLayer(parameters, self.BEFORE, context)
        after = self.parameterAsRasterLayer(parameters, self.AFTER, context)
        if before is None or after is None:
            raise QgsProcessingException("Choose both a before and an after raster.")
        name = MODELS[self.parameterAsEnum(parameters, self.MODEL, context)]
        entry = MODEL_REGISTRY[name]
        weights = resolve_weights_path(name)
        if not os.path.isfile(weights):
            raise QgsProcessingException(f"Model weights not found: {weights}. Run the installer.")
        out = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        if not out.lower().endswith(".gpkg"):
            raise QgsProcessingException("Save the changes as a GeoPackage (.gpkg).")
        work = tempfile.mkdtemp(prefix="uchange_")
        device = ("auto", "cpu", "gpu")[self.parameterAsEnum(parameters, self.DEVICE, context)]
        cmd = [VENV_PYTHON, script, "--before", before.source(), "--after", after.source(),
               "--output", work, "--output-gpkg", out, "--weights", weights, "--device", device,
               "--model-type", entry["type"], "--preset", entry["preset"], "--threshold", "auto",
               "--json-progress"]
        preset = PRESETS.get(entry["preset"], {})
        if preset.get("false_alarm_filter"):
            if self.parameterAsBool(parameters, self.FILTER, context):
                cmd += ["--filter-false-alarms", "--filter-min-score",
                        f"{self.parameterAsDouble(parameters, self.MIN_SCORE, context):.2f}"]
            else:
                cmd.append("--no-filter-false-alarms")
        if preset.get("fast_mode") and self.parameterAsBool(parameters, self.FAST, context):
            cmd.append("--fast")
        min_area = self.parameterAsDouble(parameters, self.MIN_AREA, context)
        if min_area > 0:
            cmd += ["--min-area-m2", f"{min_area:g}"]
        if parameters.get(self.EXTENT) not in (None, ""):
            r = self.parameterAsExtent(parameters, self.EXTENT, context, before.crs())
            if not r.isEmpty():
                cmd += ["--extent", f"{r.xMinimum()!r},{r.yMinimum()!r},{r.xMaximum()!r},{r.yMaximum()!r}"]
        return cmd, out, work, entry["type"] == "dinov2_lc"

    def processAlgorithm(self, parameters, context, feedback):
        cmd, out, work, labelled = self._command(parameters, context)
        err = tempfile.TemporaryFile(mode="w+")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
        lines = queue.Queue()

        def _reader():
            for ln in proc.stdout:
                lines.put(ln)
            lines.put(None)

        threading.Thread(target=_reader, daemon=True).start()
        result, error = None, None
        try:
            while True:
                if feedback.isCanceled():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    feedback.pushInfo("Stopped.")
                    return {}
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    continue
                if line is None:
                    break
                line = line.strip()
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    if line:
                        feedback.pushInfo(line)
                    continue
                kind = msg.get("type")
                if kind == "progress":
                    feedback.setProgress(msg.get("percent", 0))
                elif kind == "log" and msg.get("message"):
                    feedback.pushInfo(msg["message"])
                elif kind == "device":
                    feedback.pushInfo(f"Device: {msg.get('device', '?')}")
                elif kind == "error":
                    error = msg.get("message")
                elif kind == "result":
                    result = msg
            proc.wait()
            err.seek(0)
            stderr = err.read().strip()
        finally:
            err.close()
            if proc.poll() is None:
                proc.kill()
        if proc.returncode != 0 or error:
            raise QgsProcessingException(error or (stderr.splitlines()[-1] if stderr else "detect_changes.py failed"))
        if labelled and result:                       # land-cover rasters next to the output
            base = os.path.splitext(out)[0]
            for tag in ("from", "to"):
                src = result.get(f"{tag}_path")
                if src and os.path.isfile(src):
                    shutil.copy2(src, f"{base}_landcover_{tag}.tif")
        shutil.rmtree(work, ignore_errors=True)
        if context.willLoadLayerOnCompletion(out):
            context.layerToLoadOnCompletionDetails(out).setPostProcessor(_styler(labelled))
        return {self.OUTPUT: out}


class ChangeDetectionProvider(QgsProcessingProvider):
    def id(self):
        return "changedetection"

    def name(self):
        return "ChangeDetection"

    def icon(self):
        path = os.path.join(_HERE, "icon.png")
        return QIcon(path) if os.path.exists(path) else super().icon()

    def loadAlgorithms(self):
        self.addAlgorithm(DetectChangesAlgorithm())

    def supportedOutputVectorLayerExtensions(self):
        return ["gpkg"]

    def defaultVectorFileExtension(self, hasGeometry=True):
        return "gpkg"
