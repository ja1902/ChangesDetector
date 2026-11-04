"""Urban semantic change detection: frozen DINOv2 + one head for change and from -> to.

The head is the v0.7 change decoder (warm-started from its checkpoint) with a wider
last layer: 2 change logits, K "before" class logits and K "after" class logits,
all read from the same frozen DINOv2 features. Trained on SECOND, whose labels
exist only where something changed, so the class heads are supervised on changed
pixels only (unchanged pixels are ignored for them).

Augmentation targets what breaks on Google Earth pairs: each date gets its own
colour / brightness / season-like jitter, the second date is shifted a few pixels
(co-registration error), and scale is randomised (ground sampling distance varies).

    python train_urban_cd.py --data-root ../datasets/second/data --epochs 20
    python train_urban_cd.py --max-steps 20 --max-eval 32      # smoke test

Other datasets plug in through DATASETS: a class list and a decoder from the
dataset's label files to (change mask, before class, after class).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import uchange_qgis_plugin.model_bridge  # noqa: F401  (puts the venv on sys.path)
import cv2

IGNORE = 255
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Datasets ─────────────────────────────────────────────────────────────────
# SECOND label PNGs are RGB. White = unchanged; semantic colours follow Open-CD's
# SECOND_Dataset (opencd/datasets/second.py). "ground" is SECOND's non-vegetated
# ground surface: paved surfaces and bare land together.
SECOND_CLASSES = ("water", "ground (paved / bare)", "low vegetation", "tree",
                  "building", "sports field")
SECOND_PALETTE = ((0, 0, 255), (128, 128, 128), (0, 128, 0), (0, 255, 0),
                  (128, 0, 0), (255, 0, 0))


def decode_second(rgb):
    """(H, W, 3) label image -> (H, W) uint8: 0 = unchanged, 1..K = class + 1."""
    out = np.zeros(rgb.shape[:2], np.uint8)
    for i, colour in enumerate(SECOND_PALETTE, start=1):
        out[(rgb == colour).all(-1)] = i
    return out


DATASETS = {
    "second": dict(classes=SECOND_CLASSES, decode=decode_second,
                   dirs=("im1", "im2", "label1", "label2")),
}


def normalise(img01):
    return torch.from_numpy(((img01 - _MEAN) / _STD).transpose(2, 0, 1).copy())


class UrbanCDDataset(Dataset):
    """Pairs with (change mask, before class, after class) labels.

    Train mode returns random crops of `crop` px; eval mode returns the whole tile
    resized to `eval_size` for the images and the labels at the original size.
    """

    def __init__(self, root, split, files, dataset, train, crop=518, scale=(1.0, 1.6),
                 swap_prob=0.0, shift_px=4, photometric=1.0, eval_size=714):
        self.root, self.files, self.train = Path(root) / split, files, train
        self.spec = DATASETS[dataset]
        self.crop, self.scale, self.swap_prob = crop, scale, swap_prob
        self.shift_px, self.photometric, self.eval_size = shift_px, photometric, eval_size

    def __len__(self):
        return len(self.files)

    def _read(self, idx):
        d1, d2, l1, l2 = self.spec["dirs"]
        f = self.files[idx]
        a = np.array(Image.open(self.root / d1 / f).convert("RGB"))
        b = np.array(Image.open(self.root / d2 / f).convert("RGB"))
        la = self.spec["decode"](np.array(Image.open(self.root / l1 / f).convert("RGB")))
        lb = self.spec["decode"](np.array(Image.open(self.root / l2 / f).convert("RGB")))
        change = ((la > 0) | (lb > 0)).astype(np.uint8)
        # class maps: 0..K-1 where changed, IGNORE elsewhere
        sa = np.where(la > 0, la.astype(np.int16) - 1, IGNORE).astype(np.uint8)
        sb = np.where(lb > 0, lb.astype(np.int16) - 1, IGNORE).astype(np.uint8)
        return a, b, change, sa, sb

    def __getitem__(self, idx):
        a, b, change, sa, sb = self._read(idx)
        if not self.train:
            e = self.eval_size
            a = cv2.resize(a, (e, e), interpolation=cv2.INTER_LINEAR)
            b = cv2.resize(b, (e, e), interpolation=cv2.INTER_LINEAR)
            return (normalise(a.astype(np.float32) / 255), normalise(b.astype(np.float32) / 255),
                    torch.from_numpy(change).long(), torch.from_numpy(sa).long(),
                    torch.from_numpy(sb).long())
        rng = np.random.default_rng([torch.initial_seed() % (2 ** 32), idx])
        a, b, change, sa, sb = self._augment(rng, a, b, change, sa, sb)
        return (normalise(a), normalise(b), torch.from_numpy(change).long(),
                torch.from_numpy(sa).long(), torch.from_numpy(sb).long())

    def _augment(self, rng, a, b, change, sa, sb):
        if rng.random() < self.swap_prob:
            a, b, sa, sb = b, a, sb, sa
        # scale (the same for both dates and the labels)
        s = rng.uniform(*self.scale)
        size = max(self.crop, int(round(a.shape[0] * s)))
        a = cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (size, size), interpolation=cv2.INTER_LINEAR)
        change, sa, sb = (cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
                          for m in (change, sa, sb))
        # co-registration error: the second date moves a few pixels
        if self.shift_px and rng.random() < 0.5:
            dx, dy = (int(rng.integers(-self.shift_px, self.shift_px + 1)) for _ in range(2))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            b = cv2.warpAffine(b, M, (size, size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        # same random crop and flips / rotation for everything
        y = int(rng.integers(0, size - self.crop + 1))
        x = int(rng.integers(0, size - self.crop + 1))
        sl = (slice(y, y + self.crop), slice(x, x + self.crop))
        a, b, change, sa, sb = (m[sl] for m in (a, b, change, sa, sb))
        k = int(rng.integers(0, 4))
        flip = rng.random() < 0.5
        out = []
        for m in (a, b, change, sa, sb):
            m = np.rot90(m, k)
            out.append(np.ascontiguousarray(m[:, ::-1] if flip else m))
        a, b, change, sa, sb = out
        # each date gets its own photometric jitter (season / sensor differences)
        a = self._jitter(rng, a)
        b = self._jitter(rng, b)
        return a, b, change, sa, sb

    def _jitter(self, rng, img):
        x = img.astype(np.float32) / 255
        p = self.photometric
        if p <= 0 or rng.random() > 0.8:
            return x
        x = x * rng.uniform(1 - 0.3 * p, 1 + 0.3 * p)                      # brightness
        x = (x - x.mean()) * rng.uniform(1 - 0.3 * p, 1 + 0.3 * p) + x.mean()  # contrast
        gray = x.mean(-1, keepdims=True)
        x = gray + (x - gray) * rng.uniform(1 - 0.5 * p, 1 + 0.5 * p)      # saturation
        x = x * rng.uniform(1 - 0.1 * p, 1 + 0.1 * p, size=3)              # colour cast
        x = np.clip(x, 0, 1) ** rng.uniform(1 - 0.2 * p, 1 + 0.25 * p)     # gamma
        return np.clip(x, 0, 1).astype(np.float32)


def check_labels(root, split, files, dataset, n=100):
    """Sanity check on a sample: unknown colours, and change masks that disagree."""
    spec = DATASETS[dataset]
    d1, d2, l1, l2 = spec["dirs"]
    unknown = mismatch = total = 0
    palette = {(255, 255, 255), *SECOND_PALETTE} if dataset == "second" else None
    for f in files[:: max(1, len(files) // n)][:n]:
        ra = np.array(Image.open(Path(root) / split / l1 / f).convert("RGB"))
        rb = np.array(Image.open(Path(root) / split / l2 / f).convert("RGB"))
        la, lb = spec["decode"](ra), spec["decode"](rb)
        if palette is not None:
            known = np.zeros(ra.shape[:2], bool)
            for c in palette:
                known |= (ra == c).all(-1)
            unknown += int((~known).sum())
        mismatch += int(((la > 0) != (lb > 0)).sum())
        total += la.size
    print(f"  label check ({n} tiles): unknown colours {unknown / total:.4%} of pixels, "
          f"before/after change masks disagree on {mismatch / total:.4%}")


# ── Model ────────────────────────────────────────────────────────────────────
class UrbanHead(nn.Module):
    """The change decoder with a wider output: [change (2) | before (K) | after (K)]."""

    def __init__(self, decoder, n_classes):
        super().__init__()
        self.decoder, self.k = decoder, n_classes
        last = decoder.head[-1]
        new = nn.Conv2d(last.in_channels, 2 + 2 * n_classes, 1)
        with torch.no_grad():                    # keep the pretrained change channels
            new.weight[:2] = last.weight
            new.bias[:2] = last.bias
        decoder.head[-1] = new

    def forward(self, feats_a, feats_b, out_size):
        out = self.decoder(feats_a, feats_b, out_size=out_size)
        return out[:, :2], out[:, 2:2 + self.k], out[:, 2 + self.k:]


def build(base_ckpt, n_classes, device):
    from uchange_qgis_plugin.model_bridge import build_model
    from uchange_qgis_plugin.models.dinov2_cd import MultiLayerChangeDecoder

    backbone_model, _ = build_model(base_ckpt, device, model_type="dinov2")   # frozen, fp16
    ck = torch.load(base_ckpt, map_location="cpu", weights_only=False, mmap=True)
    cfg = ck["config"]
    decoder = MultiLayerChangeDecoder(
        in_dim=cfg["embed_dim"], hidden_dim=cfg["decoder_hidden_dim"],
        out_size=cfg["tile_size"], patch_grid=cfg["patch_grid"],
        num_layers=cfg["num_decoder_layers"])
    decoder.load_state_dict(ck["decoder"], strict=True)
    head = UrbanHead(decoder, n_classes).to(device)
    return backbone_model, head, cfg


@torch.no_grad()
def features(backbone_model, a, b):
    n = a.shape[0]
    dtype = next(backbone_model.parameters()).dtype
    feats = backbone_model.extract_multilayer_features(torch.cat([a, b]).to(dtype))
    return [f[:n].float() for f in feats], [f[n:].float() for f in feats]


def masked_ce(logits, target):
    valid = target != IGNORE
    if not bool(valid.any()):
        return logits.sum() * 0
    return F.cross_entropy(logits, target, ignore_index=IGNORE)


# ── Metrics ──────────────────────────────────────────────────────────────────
class Meter:
    """Binary change IoU/F1 and the SECOND protocol's SeK on from -> to classes."""

    def __init__(self, k, device):
        self.k = k
        self.binary = torch.zeros(4, dtype=torch.long, device=device)   # tn fp fn tp
        self.hist = torch.zeros((k + 1) ** 2, dtype=torch.long, device=device)
        self.pair_ok = self.pair_n = 0

    def update(self, p_change, p_a, p_b, change, sa, sb):
        pc, gc = p_change.bool(), change.bool()
        self.binary += torch.bincount((gc.long() * 2 + pc.long()).flatten(), minlength=4)
        k1 = self.k + 1
        for pred, gt in ((p_a, sa), (p_b, sb)):
            pred_full = torch.where(pc, pred + 1, torch.zeros_like(pred))
            gt_full = torch.where(gt == IGNORE, torch.zeros_like(gt), gt + 1)
            self.hist += torch.bincount((gt_full * k1 + pred_full).flatten(), minlength=k1 * k1)
        both = gc & (sa != IGNORE) & (sb != IGNORE)
        self.pair_ok += int((both & (p_a == sa) & (p_b == sb)).sum())
        self.pair_n += int(both.sum())

    def result(self):
        tn, fp, fn, tp = (float(v) for v in self.binary)
        iou = tp / max(tp + fp + fn, 1)
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        h = self.hist.cpu().double().numpy().reshape(self.k + 1, self.k + 1)
        c2 = np.array([[h[0, 0], h[0].sum() - h[0, 0]],
                       [h[:, 0].sum() - h[0, 0], h[1:, 1:].sum()]])
        iou2 = np.diag(c2) / np.maximum(c2.sum(0) + c2.sum(1) - np.diag(c2), 1)
        h0 = h.copy()
        h0[0, 0] = 0
        po = np.trace(h0) / h0.sum()
        pe = (h0.sum(0) * h0.sum(1)).sum() / h0.sum() ** 2
        kappa = (po - pe) / (1 - pe)
        sek = kappa * math.exp(iou2[1]) / math.e
        return dict(change_iou=iou, change_precision=prec, change_recall=rec,
                    change_f1=2 * prec * rec / max(prec + rec, 1e-9), kappa_n0=kappa,
                    sek=sek, pair_acc_on_changed=self.pair_ok / max(self.pair_n, 1))


@torch.no_grad()
def evaluate(backbone_model, head, loader, device, k, thr, max_batches=None):
    head.eval()
    meter = Meter(k, device)
    for i, (a, b, change, sa, sb) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        a, b = a.to(device), b.to(device)
        change, sa, sb = (t.to(device) for t in (change, sa, sb))
        fa, fb = features(backbone_model, a, b)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            ch, ca, cb = head(fa, fb, out_size=a.shape[2:])
        size = change.shape[1:]
        ch, ca, cb = (F.interpolate(t.float(), size=size, mode="bilinear", align_corners=False)
                      for t in (ch, ca, cb))
        p_change = torch.softmax(ch, 1)[:, 1] >= thr
        meter.update(p_change, ca.argmax(1), cb.argmax(1), change, sa, sb)
    return meter.result()


# ── Training ─────────────────────────────────────────────────────────────────
def lr_at(step, total, base, warmup=200):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", choices=list(DATASETS), default="second")
    p.add_argument("--data-root", default="../datasets/second/data",
                   help="folder with train/ and test/ (each with the dataset's image + label dirs)")
    p.add_argument("--base-ckpt", default="dinov2_vitb14_c2s1_levir.pth")
    p.add_argument("--output-dir", default="work_dirs/urban_cd")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--crop", type=int, default=518)
    p.add_argument("--scale-min", type=float, default=1.0)
    p.add_argument("--scale-max", type=float, default=1.6)
    p.add_argument("--swap-prob", type=float, default=0.0,
                   help="swap before / after (keeps classes right, drops the real-world "
                        "build-more-than-demolish prior)")
    p.add_argument("--shift-px", type=int, default=4, help="max co-registration shift to simulate")
    p.add_argument("--photometric", type=float, default=1.0, help="colour jitter strength (0 = off)")
    p.add_argument("--eval-scale", type=float, default=1.4,
                   help="tiles are upsampled by this much for evaluation (the pipeline's default is 1.4)")
    p.add_argument("--thr", type=float, default=0.5, help="change probability threshold for evaluation")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke test)")
    p.add_argument("--max-eval", type=int, default=0, help="evaluate on at most this many batches")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DATASETS[args.dataset]
    classes, k = spec["classes"], len(spec["classes"])
    root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files = sorted(os.listdir(root / "train" / spec["dirs"][0]))
    test_files = sorted(os.listdir(root / "test" / spec["dirs"][0]))
    perm = np.random.default_rng(args.seed).permutation(len(train_files))
    n_val = max(1, int(len(train_files) * args.val_frac))
    val_files = [train_files[i] for i in perm[:n_val]]
    fit_files = [train_files[i] for i in perm[n_val:]]
    print(f"{args.dataset}: {len(fit_files)} train / {len(val_files)} val / {len(test_files)} test pairs, "
          f"{k} classes: {', '.join(classes)}")
    check_labels(root, "train", train_files, args.dataset)

    eval_size = int(round(512 * args.eval_scale / 14)) * 14
    train_ds = UrbanCDDataset(root, "train", fit_files, args.dataset, True, crop=args.crop,
                              scale=(args.scale_min, args.scale_max), swap_prob=args.swap_prob,
                              shift_px=args.shift_px, photometric=args.photometric)
    val_ds = UrbanCDDataset(root, "train", val_files, args.dataset, False, eval_size=eval_size)
    test_ds = UrbanCDDataset(root, "test", test_files, args.dataset, False, eval_size=eval_size)
    mk = lambda ds, train: DataLoader(ds, batch_size=args.batch_size if train else 2, shuffle=train,
                                      num_workers=args.workers, drop_last=train, pin_memory=True)
    train_loader, val_loader, test_loader = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

    base = args.base_ckpt if os.path.isabs(args.base_ckpt) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.base_ckpt)
    backbone_model, head, cfg = build(base, k, device)
    params = [q for q in head.parameters() if q.requires_grad]
    print(f"head: {sum(q.numel() for q in params) / 1e6:.2f}M trainable params; backbone frozen")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    total = args.max_steps or args.epochs * len(train_loader)
    meta = dict(base_checkpoint=os.path.basename(base), classes=list(classes), n_classes=k,
                layer_indices=cfg["layer_indices"], embed_dim=cfg["embed_dim"],
                decoder_hidden_dim=cfg["decoder_hidden_dim"], tile_size=cfg["tile_size"],
                patch_grid=cfg["patch_grid"], num_decoder_layers=cfg["num_decoder_layers"],
                channels="change(2), before(K), after(K)", dataset=args.dataset,
                train_args=vars(args))

    def save(name, metrics):
        torch.save({"config": meta, "decoder": head.decoder.state_dict(), "metrics": metrics},
                   out_dir / name)

    step, best, t0, run = 0, -1.0, time.time(), []
    done = False
    for epoch in range(args.epochs):
        head.train()
        for a, b, change, sa, sb in train_loader:
            a, b = a.to(device), b.to(device)
            change, sa, sb = (t.to(device) for t in (change, sa, sb))
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, args.lr)
            fa, fb = features(backbone_model, a, b)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                ch, ca, cb = head(fa, fb, out_size=a.shape[2:])
            loss_c = F.cross_entropy(ch.float(), change)
            loss_a, loss_b = masked_ce(ca.float(), sa), masked_ce(cb.float(), sb)
            loss = loss_c + loss_a + loss_b
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run.append([loss_c.item(), loss_a.item(), loss_b.item()])
            step += 1
            if step % 50 == 0 or step == 1:
                m = np.mean(run[-50:], 0)
                print(f"epoch {epoch + 1} step {step}/{total}  change {m[0]:.3f}  before {m[1]:.3f}  "
                      f"after {m[2]:.3f}  ({(time.time() - t0) / step:.2f}s/step)", flush=True)
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        if done or (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            r = evaluate(backbone_model, head, val_loader, device, k, args.thr,
                         max_batches=args.max_eval or None)
            print(f"  val  IoU {r['change_iou']:.3f}  F1 {r['change_f1']:.3f}  SeK {r['sek']:.3f}  "
                  f"from->to right on changed px {r['pair_acc_on_changed']:.3f}", flush=True)
            save("urban_head_last.pth", r)
            if r["sek"] > best:
                best = r["sek"]
                save("urban_head_best.pth", r)
        if done:
            break

    print("test set (best validation checkpoint)...")
    ck = torch.load(out_dir / "urban_head_best.pth", map_location="cpu", weights_only=False)
    head.decoder.load_state_dict(ck["decoder"])
    r = evaluate(backbone_model, head, test_loader, device, k, args.thr,
                 max_batches=args.max_eval or None)
    print("  test " + "  ".join(f"{n} {v:.3f}" for n, v in r.items()))
    (out_dir / "test_metrics.json").write_text(json.dumps(r, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
