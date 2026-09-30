"""Land cover for labelled ("from -> to") change, on the DINOv2 backbone.

A small head reads the same four backbone layers as the change decoder, so
one backbone gives the change map and the land cover of each date. Trained on
OpenEarthMap (44 countries) plus SECOND; on unseen French imagery (HRSCD) it
named both the before and after class right on 50% of changed pixels, against
40% for the older SCD UPerNet (research/RESULTS_2026-09.md, E9).

Each change blob gets one label per date: the majority class inside it.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Output classes (0 = no change) are defined in model_registry, which the
# dialog imports without torch. The head predicts OpenEarthMap's classes;
# developed space and road are merged into one "paved" class.
from .model_registry import LANDCOVER_CLASSES, LANDCOVER_PALETTE  # noqa: F401

# OEM: 0 unknown 1 bareland 2 rangeland 3 developed 4 road 5 tree 6 water
# 7 agriculture 8 building
OEM_TO_LANDCOVER = np.array([0, 1, 2, 3, 3, 4, 5, 6, 7], dtype=np.uint8)
N_OEM = 9


def _cbr(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class LandCoverHead(nn.Module):
    """Multi-layer ViT tokens -> per-pixel OpenEarthMap class logits."""

    def __init__(self, in_dim=768, n_layers=4, hidden=256, n_classes=N_OEM):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(in_dim) for _ in range(n_layers)])
        self.proj = nn.ModuleList([nn.Conv2d(in_dim, hidden, 1) for _ in range(n_layers)])
        self.fuse = _cbr(hidden * n_layers, hidden)
        self.up1 = _cbr(hidden, hidden // 2)
        self.up2 = _cbr(hidden // 2, hidden // 4)
        self.cls = nn.Conv2d(hidden // 4, n_classes, 1)

    def forward(self, feats, out_size):
        maps = []
        for f, norm, proj in zip(feats, self.norms, self.proj):
            b, n, d = f.shape
            side = int(round(math.sqrt(n)))
            x = norm(f.float()).transpose(1, 2).reshape(b, d, side, n // side)
            maps.append(proj(x))
        x = self.fuse(torch.cat(maps, 1))
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = self.cls(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


def load_landcover_head(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    head = LandCoverHead(**ckpt.get("head_kwargs", {}))
    head.load_state_dict(ckpt["head"])
    return head.to(device).eval()


def _window_starts(size, win, step):
    starts = list(range(0, max(size - win, 0) + 1, step))
    if starts[-1] + win < size:
        starts.append(size - win)
    return starts


@torch.no_grad()
def predict_landcover(model, head, img, device, window=448, overlap=96, batch=8,
                      progress_fn=None):
    """uint8 HWC image -> (H, W) uint8 land-cover class (LANDCOVER_CLASSES ids).

    Sliding windows; class probabilities are averaged where windows overlap,
    one row of windows at a time so memory stays at a strip of the image.
    """
    from .model_bridge import normalize_tile

    h, w = img.shape[:2]
    win = max(14, (window // 14) * 14)
    pad_h, pad_w = max(0, win - h), max(0, win - w)
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    ph, pw = img.shape[:2]
    step = win - overlap
    ys, xs = _window_starts(ph, win, step), _window_starts(pw, win, step)
    dtype = next(model.parameters()).dtype
    out = np.zeros((ph, pw), dtype=np.uint8)

    # Rows of windows overlap by `overlap` px: keep a running strip whose top
    # part is finished once the next row starts below it.
    acc = torch.zeros(N_OEM, 0, pw)
    cnt = torch.zeros(1, 0, pw)
    top = 0                                          # image row of acc[:, 0]
    for r, y in enumerate(ys):
        need = y + win - top
        if acc.shape[1] < need:
            grow = need - acc.shape[1]
            acc = torch.cat([acc, torch.zeros(N_OEM, grow, pw)], 1)
            cnt = torch.cat([cnt, torch.zeros(1, grow, pw)], 1)
        for s in range(0, len(xs), batch):
            chunk = xs[s:s + batch]
            t = np.stack([normalize_tile(img[y:y + win, x:x + win]) for x in chunk])
            t = torch.from_numpy(t).to(device=device, dtype=dtype)
            feats = model.extract_multilayer_features(t)
            prob = torch.softmax(head(feats, (win, win)).float(), 1).cpu()
            for x, p in zip(chunk, prob):
                acc[:, y - top:y - top + win, x:x + win] += p
                cnt[:, y - top:y - top + win, x:x + win] += 1
        done = ys[r + 1] - top if r + 1 < len(ys) else acc.shape[1]
        cls = (acc[:, :done] / cnt[:, :done].clamp(min=1)).argmax(0).numpy()
        out[top:top + done] = OEM_TO_LANDCOVER[cls]
        acc, cnt, top = acc[:, done:], cnt[:, done:], top + done
        if progress_fn:
            progress_fn(r + 1, len(ys))
    return out[:h, :w]


def label_blobs(change_mask, lc_before, lc_after):
    """One (from, to) class per change blob: the majority inside it.

    Returns (blob ids int32 (H, W), from_cls, to_cls) where from_cls[k] /
    to_cls[k] belong to blob k (index 0 unused).
    """
    import cv2
    n, blobs = cv2.connectedComponents(change_mask.astype(np.uint8), connectivity=8)
    k = len(LANDCOVER_CLASSES)
    m = blobs > 0
    ids = blobs[m].astype(np.int64)

    def majority(lc):
        votes = np.bincount(ids * k + lc[m], minlength=n * k).reshape(n, k)
        votes[:, 0] = 0
        return votes.argmax(1).astype(np.uint8)

    return blobs.astype(np.int32), majority(lc_before), majority(lc_after)
