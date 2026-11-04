#!/usr/bin/env python3
"""Shrink fine-tuned checkpoints to what differs from the model they started from, and back.

A fold model of loco_finetune.py repeats the 350 MB DINOv2 backbone it started from;
only the decoder (and any unfrozen blocks) changed. slim keeps those and records the
base file in config["slim_base"]; full rebuilds a checkpoint the plugin can load.

    python benchmarks/slim_ckpt.py slim work_dirs/loco7/*/*/dinov2_finetuned.pth
    python benchmarks/slim_ckpt.py full work_dirs/loco7/unf2/r04/dinov2_finetuned.pth out.pth
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def _load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def slim(path):
    ck = _load(path)
    if "slim_base" in ck["config"]:
        return 0
    base_name = ck["config"].get("finetuned_from", "dinov2_vitb14_c2s1_levir.pth")
    base_path = Path(base_name) if Path(base_name).is_absolute() else ROOT / base_name
    base = _load(base_path)["backbone"]
    keep = {k: v for k, v in ck["backbone"].items() if k not in base or not torch.equal(v, base[k])}
    cfg = dict(ck["config"], slim_base=str(base_path))
    before = Path(path).stat().st_size
    tmp = Path(str(path) + ".tmp")
    torch.save({"backbone": keep, "decoder": ck["decoder"], "config": cfg}, tmp)
    tmp.replace(path)
    return before - Path(path).stat().st_size


def full(path, out):
    ck = _load(path)
    cfg = dict(ck["config"])
    base = cfg.pop("slim_base", None)
    bb = ck["backbone"]
    if base:
        bb = dict(_load(base)["backbone"], **bb)
    torch.save({"backbone": bb, "decoder": ck["decoder"], "config": cfg}, out)


if __name__ == "__main__":
    if sys.argv[1] == "slim":
        freed = sum(slim(p) for p in sys.argv[2:])
        print(f"freed {freed / 1e9:.2f} GB")
    else:
        full(sys.argv[2], sys.argv[3])
