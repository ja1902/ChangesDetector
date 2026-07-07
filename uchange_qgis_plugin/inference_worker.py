import traceback

import numpy as np
from qgis.core import QgsTask


class ChangeDetectionInferenceTask(QgsTask):
    def __init__(self, params):
        super().__init__("Change Detection", QgsTask.CanCancel)
        self.params = params
        self.output_path = params["output_path"]
        self.error_message = None
        self.log_messages = []

    def _log(self, msg):
        self.log_messages.append(msg)

    def run(self):
        try:
            return self._run_inference()
        except Exception as e:
            self.error_message = f"{e}\n{traceback.format_exc()}"
            self._log(f"ERROR: {e}")
            return False

    def _run_inference(self):
        from .model_bridge import _ensure_venv_on_path
        _ensure_venv_on_path()
        import torch
        from .model_bridge import build_model
        from .raster_io import read_raster, polygonize_mask
        from .tiling import run_tiled_inference

        p = self.params

        device_pref = p.get("device_preference", "auto")
        if device_pref == "cpu":
            device = torch.device("cpu")
        elif device_pref == "gpu":
            device = torch.device("cuda")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._log("Reading before raster...")
        pre_img, geotransform, projection_wkt = read_raster(p["before_path"])
        if not projection_wkt and p.get("projection_wkt"):
            projection_wkt = p["projection_wkt"]
        self.setProgress(5)

        self._log("Reading after raster...")
        post_img, _, _ = read_raster(p["after_path"])
        self.setProgress(10)

        if pre_img.shape[:2] != post_img.shape[:2]:
            self.error_message = (
                f"Image dimensions don't match: "
                f"before={pre_img.shape[:2]}, after={post_img.shape[:2]}"
            )
            return False

        self._log(f"Using device: {device}")
        self._log("Building model...")
        model, load_summary = build_model(p["checkpoint_path"], device)
        self._log(load_summary)
        self.setProgress(20)

        self._log(
            f"Running tiled inference "
            f"({pre_img.shape[1]}x{pre_img.shape[0]}, "
            f"tile={p['tile_size']}, overlap={p['overlap']})..."
        )

        def progress_fn(current, total):
            pct = 20 + int(60 * current / total)
            self.setProgress(pct)

        prob_map = run_tiled_inference(
            model, pre_img, post_img,
            tile_size=p["tile_size"],
            overlap=p["overlap"],
            device=device,
            progress_fn=progress_fn,
            cancel_fn=self.isCanceled,
        )

        if prob_map is None:
            self._log("Cancelled.")
            return False

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

        threshold = p.get("threshold", 0.5)
        binary_mask = (prob_map > threshold).astype(np.uint8)

        n_change = int(binary_mask.sum())
        self._log(f"Binary mask at threshold {threshold}: {n_change} change pixels")
        self.setProgress(85)

        self._log(f"Polygonizing to {p['output_path']}...")
        min_area = p.get("min_area", 0)

        pixel_w = abs(geotransform[1])
        pixel_h = abs(geotransform[5])
        min_area_map = min_area * pixel_w * pixel_h

        total_polys, final_polys = polygonize_mask(
            binary_mask, geotransform, projection_wkt,
            p["output_path"], min_area=min_area_map,
        )
        self._log(f"Polygons: {total_polys} created, {final_polys} after min-area filter")
        self.setProgress(100)
        self._log("Done!")
        return True

    def finished(self, result):
        pass
