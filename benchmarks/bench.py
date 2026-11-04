#!/usr/bin/env python3
"""Benchmark for speed / memory optimisations: is a change worth it?

Every optimisation is judged on the same fixed test pairs and the same scene,
against a saved run of the unchanged pipeline (the "reference").

  quality   runs the plugin's inference code (detect_changes.infer_probability,
            the same function the CLI uses) on fixed labelled test pairs and
            reports, per dataset:
              - IoU / F1 / precision / recall at the threshold the plugin ships
              - AP and best-possible IoU over all thresholds: if IoU drops but
                these hold, the change only moved the calibration and a new
                threshold (picked on other data, as for the presets) fixes it
              - agreement with the reference: mean |p - p_ref| and the share of
                pixels whose change / no-change decision flipped
              - in labelled mode, how often the (from, to) land cover of the
                reference's change pixels is unchanged
  speed     runs detect_changes.py end to end, as the plugin does, on a large
            georeferenced scene and reports wall time, inference time, peak RAM
            and peak VRAM (whole GPU, CUDA context included: what a card needs).

Typical use (options after "--" go to detect_changes.py unchanged):

  python benchmarks/bench.py quality --name baseline --save-ref
  python benchmarks/bench.py quality --name fast --ref baseline -- --fast
  python benchmarks/bench.py speed --name baseline --device cpu --scene-size 2048
  python benchmarks/bench.py speed --name fast --device cpu --scene-size 2048 -- --fast
  python benchmarks/bench.py report

Results are appended to benchmarks/results/results.jsonl (one line per run,
with the git commit, device and settings); references are kept in
benchmarks/results/refs/<name>/. A reference belongs to a device type: compare
CPU runs with a CPU reference (float32) and GPU runs with a GPU one (float16).

Test sets (fixed lists in benchmarks/sets/, made once by "make-sets"):
  s2looking  S2Looking test, 1024 px, unseen by every model, off-nadir and
             sparse change: the main verdict
  levir      LEVIR-CD test, 256 px: in-domain check
  egybcd     EGY-BCD, 256 px. Caveat: the local copy mixes its val split, on
             which the preset thresholds were picked, so read it as a
             secondary check.
Data root: --data, $CD_DATA, or ../cd-changer/open-cd-main/data.
"""
import argparse
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

RESULTS = HERE / "results"
SETS = HERE / "sets"

# name: (folder under the data root, split, before dir, after dir, label dir)
DATASETS = {
    "s2looking": ("S2Looking", "test", "Image1", "Image2", "label"),
    "levir": ("LEVIR-CD", "test", "A", "B", "label"),
    "egybcd": ("EGY-BCD", "test", "A", "B", "label"),
}
# Pairs per set: about 10 minutes for all three on a T4 with the default preset.
SET_SIZES = {"s2looking": 100, "levir": 300, "egybcd": 300}
# Small subsets for slow devices (--small): about 15 min on a 8-core desktop CPU.
SMALL_SIZES = {"s2looking": 8, "levir": 40, "egybcd": 40}

N_BINS = 2000          # probability histogram bins for AP / best IoU


def data_root(arg):
    for cand in (arg, os.environ.get("CD_DATA"), ROOT.parent / "cd-changer/open-cd-main/data"):
        if cand and Path(cand).is_dir():
            return Path(cand)
    sys.exit("Dataset folder not found: pass --data or set CD_DATA")


def pair_paths(root, name, stem):
    folder, split, a, b, lab = DATASETS[name]
    base = root / folder / split
    return base / a / f"{stem}.png", base / b / f"{stem}.png", base / lab / f"{stem}.png"


def read_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def load_set(name, small=False):
    stems = (SETS / f"{name}.txt").read_text().split()
    return stems[:SMALL_SIZES[name]] if small else stems


def git_state():
    def run(*cmd):
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "--short", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def machine(device):
    import torch
    info = {"device": device, "torch": torch.__version__, "threads": torch.get_num_threads(),
            "cpu": platform.processor() or platform.machine()}
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if hasattr(torch.backends, "cpu"):
        info["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    if device == "cuda" and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def append_result(row):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# make-sets
# ---------------------------------------------------------------------------

def cmd_make_sets(args):
    root = data_root(args.data)
    SETS.mkdir(exist_ok=True)
    for name, (folder, split, a, _b, lab) in DATASETS.items():
        stems = sorted(p.stem for p in (root / folder / split / lab).glob("*.png"))
        stems = [s for s in stems if (root / folder / split / a / f"{s}.png").is_file()]
        random.Random(0).shuffle(stems)
        chosen = stems[:SET_SIZES[name]]
        (SETS / f"{name}.txt").write_text("\n".join(chosen) + "\n")
        print(f"{name}: {len(chosen)} of {len(stems)} pairs")


# ---------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------

class Scores:
    """Pooled pixel counts over a dataset, at one threshold and over all."""

    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.hist_pos = np.zeros(N_BINS, np.int64)
        self.hist_neg = np.zeros(N_BINS, np.int64)
        self.flipped = self.pixels = 0
        self.abs_diff = 0.0
        self.worst_abs_diff = 0.0
        self.lc_same = self.lc_total = 0

    def add(self, prob, mask, label):
        self.tp += int((mask & label).sum())
        self.fp += int((mask & ~label).sum())
        self.fn += int((~mask & label).sum())
        bins = np.minimum((prob * N_BINS).astype(np.int64), N_BINS - 1)
        self.hist_pos += np.bincount(bins[label], minlength=N_BINS)
        self.hist_neg += np.bincount(bins[~label], minlength=N_BINS)
        self.pixels += mask.size

    def compare(self, prob, mask, ref_prob, ref_mask):
        d = float(np.abs(prob - ref_prob).mean())
        self.abs_diff += d * prob.size
        self.worst_abs_diff = max(self.worst_abs_diff, d)
        self.flipped += int((mask != ref_mask).sum())

    def summary(self, compared):
        tp, fp, fn = self.tp, self.fp, self.fn
        out = {"iou": tp / max(tp + fp + fn, 1),
               "f1": 2 * tp / max(2 * tp + fp + fn, 1),
               "precision": tp / max(tp + fp, 1),
               "recall": tp / max(tp + fn, 1)}
        # Threshold sweep from the histograms: predicted change = bins >= k.
        tps = np.cumsum(self.hist_pos[::-1])[::-1]
        fps = np.cumsum(self.hist_neg[::-1])[::-1]
        pos = self.hist_pos.sum()
        ious = tps / np.maximum(pos + fps, 1)
        k = int(np.argmax(ious))
        out["best_iou"], out["best_threshold"] = float(ious[k]), k / N_BINS
        prec = tps / np.maximum(tps + fps, 1)
        rec = tps / max(pos, 1)
        out["ap"] = float(np.sum((rec[:-1] - rec[1:]) * prec[:-1]) + rec[-1] * prec[-1])
        if compared:
            out["mean_abs_dp"] = self.abs_diff / max(self.pixels, 1)
            out["worst_image_abs_dp"] = self.worst_abs_diff
            out["flipped_pct"] = 100.0 * self.flipped / max(self.pixels, 1)
        if self.lc_total:
            out["landcover_same_pct"] = 100.0 * self.lc_same / self.lc_total
        return out


def pipeline_args(extra):
    """detect_changes.py settings for these options, exactly as the CLI resolves them."""
    import detect_changes as dc
    args = dc.build_parser().parse_args(["--before", "-", "--after", "-"] + list(extra))
    preset = dc.resolve_settings(args)
    if args.model_type == "dinov2_lc":
        args.mode = "binary"
    if args.mode != "binary":
        sys.exit("quality: binary and labelled change only")
    dc.resolve_threshold(args, preset)
    if args.auto_threshold:
        sys.exit("quality: per-scene auto threshold is not supported; use a preset or a number")
    if args.model_type in ("dinov2", "ensemble", "dinov2_lc"):
        args.tile_size = max(14, (args.tile_size // 14) * 14)
    return args


def cmd_quality(args):
    import torch
    import detect_changes as dc
    from uchange_qgis_plugin.cleanup import clean_mask
    from uchange_qgis_plugin.model_bridge import build_model

    root = data_root(args.data)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    p = pipeline_args(args.extra)
    labelled = p.model_type == "dinov2_lc"
    if p.max_vram and device.type == "cuda":
        from uchange_qgis_plugin.tiling import set_vram_limit
        set_vram_limit(p.max_vram, device)

    weights = p.weights if os.path.isabs(p.weights) else str(ROOT / p.weights)
    model, _ = build_model(weights, device,
                           model_type="dinov2" if p.model_type in ("ensemble", "dinov2_lc")
                           else p.model_type)
    cs_model = head = None
    if p.model_type == "ensemble":
        from uchange_qgis_plugin.changestar_bridge import build_changestar
        cs_model = build_changestar(device)[0]
    if labelled:
        from uchange_qgis_plugin.landcover import load_landcover_head
        head = load_landcover_head(str(ROOT / p.landcover_head), device)

    ref_dir = RESULTS / "refs" / args.ref if args.ref else None
    if ref_dir is not None:
        ref_meta = json.loads((ref_dir / "meta.json").read_text())
        if ref_meta["device"] != device.type:
            print(f"WARNING: reference {args.ref} was made on {ref_meta['device']}, "
                  f"this run is on {device.type}: agreement mixes in float16/float32 differences")
    save_dir = RESULTS / "refs" / args.name if args.save_ref else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    sets = args.sets.split(",")
    row = {"kind": "quality", "name": args.name, "ref": args.ref, "options": args.extra,
           "threshold": p.threshold, "small": args.small, "time": time.strftime("%Y-%m-%d %H:%M"),
           **git_state(), **machine(device.type), "sets": {}}
    for name in sets:
        stems = load_set(name, args.small)
        sc = Scores()
        t_inf = 0.0
        for i, stem in enumerate(stems):
            pa, pb, pl = pair_paths(root, name, stem)
            a, b = read_rgb(pa), read_rgb(pb)
            label = np.array(Image.open(pl).convert("L")) > 127
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            prob = dc.infer_probability(p, model, cs_model, a, b, p.scale, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_inf += time.perf_counter() - t0
            # No georeferencing here, so speckle removal by area is off (as in the CLI).
            mask = clean_mask(prob, p.threshold, p.core_threshold or None, 0, 0).astype(bool)
            sc.add(prob, mask, label)
            lc = None
            if labelled:
                blobs, f_cls, t_cls = dc.landcover_labels(model, head, device,
                                                          mask.astype(np.uint8), a, b, 1.0)
                lc = np.stack([f_cls[blobs], t_cls[blobs]])
            if ref_dir is not None:
                ref = np.load(ref_dir / name / f"{stem}.npz")
                ref_prob = ref["prob"].astype(np.float32)
                sc.compare(prob, mask, ref_prob, ref["mask"])
                if labelled and "lc" in ref:
                    keep = ref["mask"] & mask
                    same = (lc[0] == ref["lc"][0]) & (lc[1] == ref["lc"][1])
                    sc.lc_same += int((same & keep).sum())
                    sc.lc_total += int(ref["mask"].sum())
            if save_dir is not None:
                (save_dir / name).mkdir(exist_ok=True)
                extra = {"lc": lc} if lc is not None else {}
                np.savez_compressed(save_dir / name / f"{stem}.npz",
                                    prob=prob.astype(np.float16), mask=mask, **extra)
            print(f"\r{name}: {i + 1}/{len(stems)}", end="", flush=True)
        s = sc.summary(ref_dir is not None)
        s.update(n=len(stems), seconds=t_inf, seconds_per_pair=t_inf / max(len(stems), 1))
        row["sets"][name] = s
        print("\r" + format_quality(name, s), flush=True)

    if save_dir is not None:
        (save_dir / "meta.json").write_text(json.dumps(
            {"device": device.type, "options": args.extra, "small": args.small,
             "sets": sets, **git_state()}, indent=1))
    append_result(row)


def format_quality(name, s):
    txt = (f"{name:10s} n={s['n']:<4d} IoU {s['iou']:.4f}  F1 {s['f1']:.4f}  "
           f"P {s['precision']:.3f} R {s['recall']:.3f}  AP {s['ap']:.4f}  "
           f"best IoU {s['best_iou']:.4f} @ {s['best_threshold']:.3f}  "
           f"{s['seconds_per_pair']:.2f} s/pair")
    if "mean_abs_dp" in s:
        txt += f"  |dp| {s['mean_abs_dp']:.5f}  flipped {s['flipped_pct']:.3f}%"
    if "landcover_same_pct" in s:
        txt += f"  land cover same {s['landcover_same_pct']:.2f}%"
    return txt


# ---------------------------------------------------------------------------
# speed
# ---------------------------------------------------------------------------

def make_scene(root, size):
    """A size x size georeferenced pair (0.5 m, UTM) tiled from S2Looking test pairs."""
    out = RESULTS / "scenes" / f"s2looking_{size}"
    paths = [out / "before.tif", out / "after.tif"]
    if all(p.is_file() for p in paths):
        return paths
    from osgeo import gdal, osr
    out.mkdir(parents=True, exist_ok=True)
    stems = load_set("s2looking")
    n = -(-size // 1024)
    if n * n > len(stems):
        sys.exit("scene too large for the S2Looking set")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32650)
    for k, path in enumerate(paths):
        ds = gdal.GetDriverByName("GTiff").Create(str(path), size, size, 3, gdal.GDT_Byte,
                                                  ["COMPRESS=DEFLATE", "TILED=YES"])
        ds.SetGeoTransform((400000.0, 0.5, 0.0, 3500000.0, 0.0, -0.5))
        ds.SetProjection(srs.ExportToWkt())
        for i in range(n * n):
            y, x = divmod(i, n)
            img = read_rgb(pair_paths(root, "s2looking", stems[i])[k])
            h, w = min(1024, size - y * 1024), min(1024, size - x * 1024)
            for c in range(3):
                ds.GetRasterBand(c + 1).WriteArray(np.ascontiguousarray(img[:h, :w, c]),
                                                   x * 1024, y * 1024)
        ds = None
    return paths


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                              "-i", "0"], capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def cmd_speed(args):
    root = data_root(args.data)
    before, after = make_scene(root, args.scene_size)
    out_dir = RESULTS / "speed_out"
    cmd = [sys.executable, str(ROOT / "detect_changes.py"), "--before", str(before),
           "--after", str(after), "--no-coreg", "--json-progress", "--device", args.device,
           "--output", str(out_dir), "--output-gpkg", str(out_dir / "changes.gpkg")] + args.extra
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for r in range(args.repeat):
        if (out_dir / "changes.gpkg").exists():
            (out_dir / "changes.gpkg").unlink()
        idle = gpu_used_mb() if args.device != "cpu" else None
        peak = [idle]
        stop = threading.Event()

        def poll():
            while not stop.wait(0.1):
                used = gpu_used_mb()
                if used is not None and (peak[0] is None or used > peak[0]):
                    peak[0] = used

        poller = threading.Thread(target=poll, daemon=True)
        if idle is not None:
            poller.start()
        log_out, log_err = out_dir / "stdout.log", out_dir / "stderr.log"
        with open(log_out, "w") as fo, open(log_err, "w") as fe:
            t0 = time.perf_counter()
            proc = subprocess.Popen(cmd, stdout=fo, stderr=fe)
            # wait4: this child's own peak resident memory (ru_maxrss, kB on Linux)
            _, status, usage = os.wait4(proc.pid, 0)
            wall = time.perf_counter() - t0
        proc.returncode = os.waitstatus_to_exitcode(status)
        rss_mb = usage.ru_maxrss / 1024.0
        stop.set()
        stdout, stderr = log_out.read_text(), log_err.read_text()
        if proc.returncode != 0:
            print(stdout[-3000:], stderr[-3000:])
            sys.exit("detect_changes.py failed")
        inference = None
        for line in stdout.splitlines():
            if line.startswith("{"):
                msg = json.loads(line).get("message", "")
                if msg.startswith("Inference: "):
                    inference = float(msg.split()[1].rstrip("s"))
        run = {"wall_s": wall, "inference_s": inference, "peak_rss_mb": rss_mb}
        if idle is not None:
            run["peak_vram_mb"] = (peak[0] or 0) - idle
        runs.append(run)
        print(f"run {r + 1}: wall {wall:.1f} s, inference {inference} s, "
              f"peak RAM {rss_mb:.0f} MB"
              + (f", peak VRAM {run['peak_vram_mb']:.0f} MB" if "peak_vram_mb" in run else ""),
              flush=True)
    import torch
    row = {"kind": "speed", "name": args.name, "options": args.extra,
           "scene": f"s2looking_{args.scene_size}", "time": time.strftime("%Y-%m-%d %H:%M"),
           **git_state(), **machine("cuda" if args.device != "cpu" and torch.cuda.is_available()
                                    else "cpu"),
           "runs": runs,
           "best_wall_s": min(r["wall_s"] for r in runs),
           "best_inference_s": min(r["inference_s"] or 1e9 for r in runs),
           "peak_rss_mb": max(r["peak_rss_mb"] for r in runs)}
    if any("peak_vram_mb" in r for r in runs):
        row["peak_vram_mb"] = max(r.get("peak_vram_mb", 0) for r in runs)
    append_result(row)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(args):
    path = RESULTS / "results.jsonl"
    if not path.is_file():
        sys.exit("no results yet")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    latest = {}
    for r in rows:                      # last run of each (kind, name, device, size)
        key = (r["kind"], r["name"], r["device"], r.get("small"), r.get("scene"))
        latest[key] = r
    print("QUALITY  (IoU at the shipped threshold; AP and best IoU over thresholds; "
          "agreement with the reference)")
    for (kind, name, device, small, _), r in latest.items():
        if kind != "quality":
            continue
        print(f"\n[{name}] {device}{' small' if small else ''}  options: "
              f"{' '.join(r['options']) or '(defaults)'}  ref: {r['ref'] or '-'}  "
              f"@{r['commit']}{'+' if r['dirty'] else ''}")
        for set_name, s in r["sets"].items():
            print("  " + format_quality(set_name, s))
    print("\nSPEED  (end to end detect_changes.py)")
    for (kind, name, device, _, scene), r in latest.items():
        if kind != "speed":
            continue
        vram = f"  VRAM {r['peak_vram_mb']:.0f} MB" if "peak_vram_mb" in r else ""
        print(f"  {name:20s} {device:4s} {scene:16s} wall {r['best_wall_s']:7.1f} s  "
              f"inference {r['best_inference_s']:7.1f} s  RAM {r['peak_rss_mb']:.0f} MB{vram}  "
              f"options: {' '.join(r['options']) or '(defaults)'}")


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=None, help="dataset root (S2Looking/, LEVIR-CD/, EGY-BCD/)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make-sets", help="pick the fixed test pairs (once; commit the lists)")
    q = sub.add_parser("quality", help="accuracy and agreement on the fixed test pairs")
    q.add_argument("--name", required=True)
    q.add_argument("--ref", default=None, help="reference run to compare with")
    q.add_argument("--save-ref", action="store_true", help="save this run as a reference")
    q.add_argument("--device", default="auto")
    q.add_argument("--sets", default="s2looking,levir,egybcd")
    q.add_argument("--small", action="store_true", help="small subsets, for CPU runs")
    s = sub.add_parser("speed", help="end-to-end time and memory on a large scene")
    s.add_argument("--name", required=True)
    s.add_argument("--device", default="auto")
    s.add_argument("--scene-size", type=int, default=2048, help="scene side in pixels")
    s.add_argument("--repeat", type=int, default=1)
    sub.add_parser("report", help="latest result of every run")
    args = ap.parse_args(argv)
    args.extra = extra
    {"make-sets": cmd_make_sets, "quality": cmd_quality, "speed": cmd_speed,
     "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
