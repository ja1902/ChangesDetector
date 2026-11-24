#!/usr/bin/env python3
"""Leave-one-city-out: does fine-tuning on your squares help in a city it never saw?

For each city in turn: fine-tune the shipped model on the squares of the other
cities (finetune_dinov2.py), run it on the held-out city's squares and compare with
the shipped model there. A city is the first three letters of a square's name, so a
square and its sibling (xyz_c1, xyzb_c1) are held out together. Two kinds of score on held-out squares:
  objects   recall of the reviewed real changes and false alarms per km2, at the
            preset threshold (score_testset.py)
  pixels    AP and best IoU against train_label.tif (SAM outlines; ignore areas left out)

    python benchmarks/loco_finetune.py --squares work_dirs/train_squares
    python benchmarks/loco_finetune.py ... --hold r04 --steps 300        # one fold, quick
    python benchmarks/loco_finetune.py ... --name ft_hn --out work_dirs/loco7/hn -- --hard-neg 2
                                         # a technique: arguments after -- go to finetune_dinov2.py

Caveat: the reviewed changes come from the shipped model, the ensemble and the
nudged model, so a change only the fine-tuned model finds counts against it as a
false alarm until it is added to the square's "missed" points.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from osgeo import gdal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import uchange_qgis_plugin.model_bridge  # noqa: E402,F401
from bench import Scores, pipeline_args  # noqa: E402
from candidates import save_prob  # noqa: E402
from score_testset import score_pair  # noqa: E402
from slim_ckpt import slim  # noqa: E402

gdal.UseExceptions()


def infer(weights, folder, device):
    import detect_changes as dc
    from uchange_qgis_plugin.model_bridge import build_model, _model_cache
    from uchange_qgis_plugin.resolution import working_factor
    p = pipeline_args(["--weights", str(weights), "--preset", "synthetic"])
    if "slim_base" in torch.load(weights, map_location="cpu", weights_only=False, mmap=True)["config"]:
        import tempfile
        from slim_ckpt import full
        tmp = Path(tempfile.mkdtemp(dir=Path(weights).parent)) / "full.pth"
        full(weights, tmp)                           # a slimmed fold model: rebuild it first
        model, _ = build_model(str(tmp), device, model_type="dinov2")
        tmp.unlink()
        tmp.parent.rmdir()
    else:
        model, _ = build_model(str(weights), device, model_type="dinov2")
    a, geo = dc.read_image(str(folder / "before.tif"))
    b, _ = dc.read_image(str(folder / "after.tif"))
    factor, _, _ = working_factor(geo, p.target_gsd, p.scale, a.shape)
    prob = dc.infer_probability(p, model, None, a, b, factor, device)
    _model_cache.clear()                             # build_model keeps every model it loads
    del model
    torch.cuda.empty_cache()
    return prob, geo, p.threshold


def pixel_scores(folder, names):
    lab = gdal.Open(str(folder / "train_label.tif")).ReadAsArray()
    valid = lab != 255
    out = {}
    for n in names:
        p = gdal.Open(str(folder / f"prob_{n}.tif")).ReadAsArray().astype(np.float32) / 255
        sc = Scores()
        sc.add(p[valid], p[valid] >= 0.5, lab[valid] == 1)
        s = sc.summary(False)
        out[n] = {"ap": s["ap"], "best_iou": s["best_iou"]}
    return out


def city(square):
    return square[:3]


def main():
    argv = sys.argv[1:]
    ft_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squares", required=True)
    ap.add_argument("--hold", default=None, help="comma list of cities to hold out (default: every city)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--name", default="loco", help="the held-out probabilities go to prob_<name>.tif")
    ap.add_argument("--out", default="work_dirs/loco")
    args = ap.parse_args(argv)

    sq = Path(args.squares)
    labelled = sorted(p.name for p in sq.iterdir() if (p / "train_label.tif").is_file())
    cities = sorted({city(s) for s in labelled})
    holds = args.hold.split(",") if args.hold else cities
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nm = args.name
    thr = {"default": pipeline_args([]).threshold}
    results = {}
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if (out_root / "results.json").is_file():           # resume: keep finished folds
        results = json.loads((out_root / "results.json").read_text())
    for c in holds:
        test = [s for s in labelled if city(s) == c]
        train = [s for s in labelled if city(s) != c]
        if all(s in results for s in test):
            print(f"=== {c}: done already", flush=True)
            continue
        out = out_root / c
        print(f"\n=== hold out {c} ({', '.join(test)}); train on {', '.join(train)}", flush=True)
        t0 = time.time()
        if not (out / "dinov2_finetuned.pth").is_file():
            subprocess.run([sys.executable, str(ROOT / "finetune_dinov2.py"), "--squares", str(sq),
                            "--only", ",".join(train), "--steps", str(args.steps), "--out", str(out),
                            *ft_args], check=True, cwd=ROOT)
        for h in test:
            prob, geo, thr[nm] = infer(out / "dinov2_finetuned.pth", sq / h, device)
            save_prob(sq / h / f"prob_{nm}.tif", prob, geo)
            res, km2 = score_pair(sq / h, ["default", nm], thr, 3.0, 50.0)
            pix = pixel_scores(sq / h, ["default", nm])
            row = {}
            for d in ("default", nm):
                f = sum(v[0] for v in res[d]["found"].values())
                n = sum(v[1] for v in res[d]["found"].values())
                row["default" if d == "default" else "loco"] = {
                    "recall": f / max(n, 1), "found": f, "changes": n,
                    "false_alarms_km2": res[d]["false_alarms"] / km2, **pix[d]}
            results[h] = row
            print(f"{h} ({time.time() - t0:.0f}s):", flush=True)
            for d, r in row.items():
                print(f"  {'shipped' if d == 'default' else 'fine-tuned':10s} recall {r['recall']:.3f} ({r['found']}/{r['changes']})  "
                      f"false alarms {r['false_alarms_km2']:5.1f}/km2  pixel AP {r['ap']:.3f}  best IoU {r['best_iou']:.3f}",
                      flush=True)
        (out_root / "results.json").write_text(json.dumps(results, indent=1))
        slim(out / "dinov2_finetuned.pth")     # keep only what differs from the shipped file
    print("\n=== summary (held-out squares)")
    for d in ("default", "loco"):
        f = sum(r[d]["found"] for r in results.values())
        n = sum(r[d]["changes"] for r in results.values())
        fa = np.mean([r[d]["false_alarms_km2"] for r in results.values()])
        apm = np.mean([r[d]["ap"] for r in results.values()])
        iou = np.mean([r[d]["best_iou"] for r in results.values()])
        print(f"  {'shipped' if d == 'default' else 'fine-tuned':10s} recall {f / max(n, 1):.3f} ({f}/{n})  "
              f"false alarms {fa:5.1f}/km2  pixel AP {apm:.3f}  best IoU {iou:.3f}")


if __name__ == "__main__":
    main()
