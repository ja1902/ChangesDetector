#!/usr/bin/env python3
"""Fine-tune a DINOv2 change model on your own labelled squares, mixed with LEVIR-CD.

Starts from a plugin checkpoint (default: the shipped dinov2_vitb14_c2s1_levir.pth).
DINOv2 stays frozen; only the decoder learns. Every batch is half your squares
(benchmarks/make_labels.py: train_label.tif with 0 / 1 / 255 = ignore) and half LEVIR-CD,
so the model learns your imagery without forgetting what it knew.

Your squares are sampled as random crops at the scale the plugin runs them at
(x1.4, and up to 0.5 m pixels first if coarser), half of them centred on a change,
with flips, 90-degree turns, a separate colour jitter per date and random date order.

    python finetune_dinov2.py --squares work_dirs/train_squares --levir ../cd-changer/open-cd-main/data/LEVIR-CD
    python finetune_dinov2.py ... --steps 50          # smoke test
    python finetune_dinov2.py ... --hard-frac 0.3 --hard-weight 2
                     # hard negatives: 30% of crops centred on candidates you marked
                     # "false", and their pixels count twice in the loss
    python finetune_dinov2.py ... --unfreeze 2       # also train DINOv2's last 2 blocks (lr x0.1)
    python finetune_dinov2.py ... --batch 3 --levir-batch 1   # 25% LEVIR instead of 50%

Writes <out>/dinov2_finetuned.pth in the plugin's format (backbone + decoder + config):
load it in QGIS or pass it to detect_changes.py / benchmarks/bench.py with --weights.
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from osgeo import gdal
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import uchange_qgis_plugin.model_bridge  # noqa: E402,F401
from train_dinov2_cd import CDDataset  # noqa: E402

gdal.UseExceptions()
IGNORE = 255


def false_mask(folder):
    """Pixels of the candidates (both rounds) marked "false"."""
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from score_testset import burn
    ds = gdal.Open(str(folder / "before.tif"))
    m = np.zeros((ds.RasterYSize, ds.RasterXSize), bool)
    for gpkg, layer in ((folder / "candidates.gpkg", "candidates"), (folder / "extra.gpkg", "extra")):
        if gpkg.is_file():
            m |= burn(ds, gpkg, layer, "verdict = 'false'")[0] > 0
    return m
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def norm(img):
    return torch.from_numpy(((img.astype(np.float32) / 255 - _MEAN) / _STD).transpose(2, 0, 1).copy())


def jitter(img, rng):
    """Brightness / contrast / saturation / tint, one draw per date."""
    x = img.astype(np.float32)
    x = x * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    mean = x.mean((0, 1), keepdims=True)
    x = (x - mean) * rng.uniform(0.75, 1.25) + mean
    g = x.mean(2, keepdims=True)
    x = (x - g) * rng.uniform(0.6, 1.4) + g
    x = x * rng.uniform(0.92, 1.08, size=(1, 1, 3))
    return np.clip(x, 0, 255).astype(np.uint8)


def misalign_crop(img, y, x, c, amp_px, rng):
    """c x c crop at (y, x) of img, resampled through a random smooth displacement field:
    a global shift plus a gentle local wobble, together at most ~amp_px pixels per axis.
    Mimics the small, uneven co-registration error of off-nadir pairs."""
    import cv2
    h, w = img.shape[:2]
    m = int(np.ceil(amp_px)) + 2
    y0, x0, y1, x1 = max(0, y - m), max(0, x - m), min(h, y + c + m), min(w, x + c + m)
    win = np.ascontiguousarray(img[y0:y1, x0:x1])
    glob = rng.uniform(-0.6, 0.6, size=2) * amp_px
    loc = rng.uniform(-0.4, 0.4, size=(3, 3, 2)).astype(np.float32) * amp_px
    field = cv2.resize(loc, (c, c), interpolation=cv2.INTER_CUBIC) + glob.astype(np.float32)
    gx, gy = np.meshgrid(np.arange(c, dtype=np.float32), np.arange(c, dtype=np.float32))
    mx = gx + (x - x0) + field[..., 0]
    my = gy + (y - y0) + field[..., 1]
    return cv2.remap(win, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def shadow_jitter(img, rng, px_m):
    """Shadows that differ between dates: either soft dark patches anywhere, or the
    image's own dark areas (mostly shadows) partly lifted and re-cast a few metres
    away, as a different sun position would."""
    import cv2
    x = img.astype(np.float32)
    h, w = x.shape[:2]
    if rng.random() < 0.5:
        noise = cv2.resize(rng.random((6, 6)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        mask = (noise > np.quantile(noise, rng.uniform(0.7, 0.9))).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
        x *= 1 - rng.uniform(0.3, 0.6) * mask[..., None]
    else:
        lum = x.mean(2)
        dark = (lum < np.percentile(lum, rng.uniform(10, 20))).astype(np.float32)
        dark = cv2.GaussianBlur(dark, (0, 0), 1.5)
        d = rng.uniform(2, 8) / px_m
        ang = rng.uniform(0, 2 * np.pi)
        M = np.float32([[1, 0, d * np.cos(ang)], [0, 1, d * np.sin(ang)]])
        cast = cv2.warpAffine(dark, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        x *= (1 + rng.uniform(0.3, 0.8) * dark)[..., None]          # lift the old shadows
        x *= (1 - rng.uniform(0.3, 0.6) * np.clip(cast - dark, 0, 1))[..., None]   # cast new ones
    return np.clip(x, 0, 255).astype(np.uint8)


class Squares(torch.utils.data.Dataset):
    """Random crops of the labelled squares, at the plugin's working scale."""

    def __init__(self, folders, crop=518, scale=1.4, target_gsd=0.5, length=100000, hard_frac=0.0,
                 misalign_m=0.0, shadow_p=0.0):
        self.crop, self.length, self.hard_frac = crop, length, hard_frac
        self.misalign_m, self.shadow_p = misalign_m, shadow_p
        self.items = []
        for f in folders:
            ds = gdal.Open(str(f / "before.tif"))
            gsd = abs(ds.GetGeoTransform()[1])
            factor = scale * (gsd / target_gsd if gsd > target_gsd * 1.05 else 1.0)
            a = ds.ReadAsArray()[:3].transpose(1, 2, 0)
            b = gdal.Open(str(f / "after.tif")).ReadAsArray()[:3].transpose(1, 2, 0)
            lab = gdal.Open(str(f / "train_label.tif")).ReadAsArray()
            size = (int(round(a.shape[1] * factor)), int(round(a.shape[0] * factor)))
            a = np.array(Image.fromarray(a).resize(size, Image.BILINEAR))
            b = np.array(Image.fromarray(b).resize(size, Image.BILINEAR))
            lab = np.array(Image.fromarray(lab).resize(size, Image.NEAREST))
            ys, xs = np.nonzero(lab == 1)
            hard = np.zeros(lab.shape, bool)
            if hard_frac > 0:
                hard = np.array(Image.fromarray(false_mask(f).astype(np.uint8)).resize(size, Image.NEAREST)) > 0
                hard &= lab == 0                     # a drawing or an outline band wins
            self.items.append((f.name, a, b, lab, ys, xs, hard, *np.nonzero(hard), gsd / factor))

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        rng = np.random.default_rng(int(torch.randint(1 << 62, (1,))))   # differs per worker and call
        _, a, b, lab, ys, xs, hard, hy, hx, px_m = self.items[rng.integers(len(self.items))]
        h, w = lab.shape
        c = self.crop
        u = rng.random()
        if len(hy) and u < self.hard_frac:          # centred on a false alarm you rejected
            k = rng.integers(len(hy))
            y = int(np.clip(hy[k] - rng.integers(c), 0, h - c))
            x = int(np.clip(hx[k] - rng.integers(c), 0, w - c))
        elif len(ys) and rng.random() < 0.5:
            k = rng.integers(len(ys))
            y = int(np.clip(ys[k] - rng.integers(c), 0, h - c))
            x = int(np.clip(xs[k] - rng.integers(c), 0, w - c))
        else:
            y, x = int(rng.integers(h - c + 1)), int(rng.integers(w - c + 1))
        ca, cb, cl = a[y:y + c, x:x + c], b[y:y + c, x:x + c], lab[y:y + c, x:x + c]
        ch = hard[y:y + c, x:x + c]
        if self.misalign_m > 0:                      # labels stay in the before image's frame
            cb = misalign_crop(b, y, x, c, rng.uniform(0, 1) * self.misalign_m / px_m, rng)
        if self.shadow_p > 0:
            if rng.random() < self.shadow_p:
                cb = shadow_jitter(cb, rng, px_m)
            if rng.random() < self.shadow_p / 2:
                ca = shadow_jitter(ca, rng, px_m)
        k = int(rng.integers(4))
        ca, cb, cl, ch = np.rot90(ca, k), np.rot90(cb, k), np.rot90(cl, k), np.rot90(ch, k)
        if rng.random() < 0.5:
            ca, cb, cl, ch = ca[:, ::-1], cb[:, ::-1], cl[:, ::-1], ch[:, ::-1]
        ca, cb = jitter(ca, rng), jitter(cb, rng)
        if rng.random() < 0.5:                       # change is symmetric in time
            ca, cb = cb, ca
        return (norm(ca), norm(cb), torch.from_numpy(cl.astype(np.int64).copy()),
                torch.from_numpy(ch.copy()))


def forward_partial(model, a, b, n_free):
    """The detector's forward, with gradients through DINOv2's last n_free blocks."""
    bb = model.backbone
    cut = len(bb.blocks) - n_free
    x = torch.cat([a, b])
    feats = []
    with torch.no_grad():
        x = bb.patch_embed(x)
        pos = bb._interpolate_pos_embed(x.shape[1])
        x = torch.cat([(bb.cls_token + pos[:, :1]).expand(x.shape[0], -1, -1), x + pos[:, 1:]], dim=1)
        x = bb.pos_drop(x)
        for i in range(cut):
            x = bb.blocks[i](x)
            if i in model.layer_indices:
                feats.append(x[:, 1:])
    for i in range(cut, len(bb.blocks)):
        x = bb.blocks[i](x)
        if i in model.layer_indices:
            feats.append(x[:, 1:])
    n = a.shape[0]
    return model.decoder([f[:n] for f in feats], [f[n:] for f in feats], out_size=a.shape[2:])


def endless(loader):
    while True:
        yield from loader


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squares", required=True, help="folder of labelled squares (train_label.tif in each)")
    ap.add_argument("--only", default=None, help="comma list of squares to train on (default: all)")
    ap.add_argument("--levir", default="../cd-changer/open-cd-main/data/LEVIR-CD")
    ap.add_argument("--weights", default="dinov2_vitb14_c2s1_levir.pth")
    ap.add_argument("--out", default="work_dirs/finetune_gee")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2, help="crops of your squares per step")
    ap.add_argument("--levir-batch", type=int, default=None, help="LEVIR crops per step (default: --batch)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hard-frac", type=float, default=0.0,
                    help="share of your crops centred on candidates marked false")
    ap.add_argument("--hard-weight", type=float, default=1.0,
                    help="loss weight of the pixels of candidates marked false")
    ap.add_argument("--unfreeze", type=int, default=0, help="also train DINOv2's last N blocks")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--misalign-m", type=float, default=0.0,
                    help="warp the after crop by a random smooth shift of up to this many metres (0 = off)")
    ap.add_argument("--shadow-p", type=float, default=0.0,
                    help="chance of a shadow change on the after crop (half that on the before crop; 0 = off)")
    args = ap.parse_args()

    from uchange_qgis_plugin.models import build_dinov2_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)
    model, _ = build_dinov2_model(str(ROOT / args.weights), device)
    model.float()                                   # decoder trains in float32 (autocast below)
    model.backbone.half()
    free = list(model.backbone.blocks[len(model.backbone.blocks) - args.unfreeze:]) if args.unfreeze else []
    for blk in free:
        blk.float()
        for q in blk.parameters():
            q.requires_grad = True
    n_lev = args.levir_batch or args.batch
    share = args.batch / (args.batch + n_lev)       # loss weight of your squares
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    folders = sorted(p for p in Path(args.squares).iterdir() if (p / "train_label.tif").is_file())
    if args.only:
        folders = [p for p in folders if p.name in args.only.split(",")]
    if not folders:
        sys.exit("no labelled squares (run benchmarks/make_labels.py first)")
    print("your squares:", ", ".join(p.name for p in folders))
    own = torch.utils.data.DataLoader(Squares(folders, hard_frac=args.hard_frac, misalign_m=args.misalign_m,
                                              shadow_p=args.shadow_p), batch_size=args.batch,
                                      num_workers=args.workers)
    lev = torch.utils.data.DataLoader(CDDataset(args.levir, "train", 518, augment=True),
                                      batch_size=n_lev, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
    own_it, lev_it = endless(own), endless(lev)

    groups = [{"params": list(model.decoder.parameters()), "lr": args.lr}]
    if free:
        groups.append({"params": [q for blk in free for q in blk.parameters()],
                       "lr": args.lr * args.backbone_lr_mult})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=args.steps,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    t0, run = time.time(), {"own": 0.0, "levir": 0.0}
    for step in range(1, args.steps + 1):
        oa, ob, ol, oh = next(own_it)
        la, lb, ll = next(lev_it)
        a = torch.cat([oa, la]).to(device).half()
        b = torch.cat([ob, lb]).to(device).half()
        lab = torch.cat([ol, ll]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = forward_partial(model, a, b, args.unfreeze) if free else model(a, b)
        logits = logits.float()
        n = len(oa)
        if args.hard_weight != 1.0:
            ce = F.cross_entropy(logits[:n], lab[:n], ignore_index=IGNORE, reduction="none")
            wt = (1 + (args.hard_weight - 1) * oh.to(device).float()) * (lab[:n] != IGNORE)
            l_own = (ce * wt).sum() / wt.sum().clamp(min=1)
        else:
            l_own = F.cross_entropy(logits[:n], lab[:n], ignore_index=IGNORE)
        l_lev = F.cross_entropy(logits[n:], lab[n:], ignore_index=IGNORE)
        loss = share * l_own + (1 - share) * l_lev
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run["own"] += l_own.item()
        run["levir"] += l_lev.item()
        if step % 50 == 0 or step == args.steps:
            k = 50 if step % 50 == 0 else step % 50
            print(f"step {step}/{args.steps}  loss yours {run['own'] / k:.4f}  LEVIR {run['levir'] / k:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  ({time.time() - t0:.0f}s)", flush=True)
            run = {"own": 0.0, "levir": 0.0}
    cfg = dict(ck["config"], finetuned_from=args.weights, finetune_squares=[p.name for p in folders],
               finetune_steps=args.steps, finetune_hard=[args.hard_frac, args.hard_weight],
               finetune_unfreeze=args.unfreeze, finetune_levir_batch=n_lev,
               finetune_aug=[args.misalign_m, args.shadow_p])
    path = out / "dinov2_finetuned.pth"
    backbone = ck["backbone"]
    if free:
        cut = len(model.backbone.blocks) - args.unfreeze      # only the trained blocks change
        backbone = {k: (v.float().cpu() if k.startswith("blocks.") and int(k.split(".")[1]) >= cut
                        else ck["backbone"][k]) for k, v in model.backbone.state_dict().items()
                    if k in ck["backbone"]}
    torch.save({"backbone": backbone, "decoder": model.decoder.state_dict(), "config": cfg}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
