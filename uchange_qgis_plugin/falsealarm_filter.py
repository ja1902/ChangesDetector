"""False-alarm filter for change polygons (finetuned model).

The change map is thresholded low, and each blob gets a second opinion: a small
classifier trained on our real / false verdicts looks at frozen DINOv2 features of the
blob and a ring around it, on both dates, together with the map's own confidence in the
blob. Blobs judged false (mostly building lean and shadows) are dropped. Each kept blob
is then redrawn as its part above the usual threshold, so outlines stay as tight as
without the filter.

Tested on the 8 held-out squares of our dataset (each scored by a model and classifier
trained without its place) and on pr2_c1, which neither had seen
(work_dirs/exp_recheck/polyfilter.py, polyfilter_fresh.py): at the same share of changes
found, 20-45 % fewer false alarms; on pr2 16/23 found with 14 false alarms per km2,
against 28 without the filter. It can drop a real change that touches a big new shadow,
because the low threshold merges the two into one blob.

Only valid on the map it was trained on: the finetuned model at x1.4 with flip
averaging and WITHOUT the label-free nudge (preset "synthetic" pipeline). The backbone is
frozen and the same in every DINOv2 checkpoint here, so the change model's backbone is
reused. Weights: weights/falsealarm_filter_finetuned.npz (work_dirs/exp_recheck/export_filter.py).
"""
import numpy as np
import torch
from PIL import Image

MIN_BLOB_M2 = 50.0        # smaller blobs were never judged (nor scored)
CONTEXT_M, MIN_HALF_M = 15.0, 20.0
RING_M = 8.0

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_filter(path):
    f = np.load(path)
    return {k: f[k] for k in f.files}


def _crop(img, y0, y1, x0, x1):
    """img[y0:y1, x0:x1] with edge padding where the window leaves the image."""
    H, W = img.shape[:2]
    c = img[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    pad = ((max(0, -y0), max(0, y1 - H)), (max(0, -x0), max(0, x1 - W)), (0, 0))
    return np.pad(c, pad, mode="edge") if any(p for pp in pad for p in pp) else c


class _Features:
    def __init__(self, backbone, device, size, layers):
        self.bb, self.dev, self.size, self.layers = backbone, device, int(size), tuple(int(l) for l in layers)
        self.dtype = next(backbone.parameters()).dtype

    @torch.no_grad()
    def _tokens(self, x):
        bb, g = self.bb, self.size // 14
        x = bb.patch_embed(x)
        pe = bb._interpolate_pos_embed(x.shape[1])
        x = x + pe[:, 1:]
        x = torch.cat([(bb.cls_token + pe[:, :1]).expand(x.shape[0], -1, -1), x], 1)
        out = []
        for i, blk in enumerate(bb.blocks):
            x = blk(x)
            if i in self.layers:
                out.append(x[:, 1:].reshape(x.shape[0], g, g, -1))
            if i >= max(self.layers):
                break
        return out

    def _prep(self, im):
        t = np.asarray(Image.fromarray(np.ascontiguousarray(im[..., :3])).resize((self.size, self.size), Image.LANCZOS))
        return np.transpose((t.astype(np.float32) / 255 - _MEAN) / _STD, (2, 0, 1))

    @torch.no_grad()
    def blob(self, A, B, ys, xs, gsd):
        """(L, 2 in/ring, 2 dates, 768) for one blob, averaged over 4 rotations."""
        from scipy import ndimage as ndi
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        h = max(max(y1 - y0, x1 - x0) // 2 + int(CONTEXT_M / gsd), int(MIN_HALF_M / gsd))
        mask = np.zeros((2 * h, 2 * h), bool)
        my, mx = ys - (cy - h), xs - (cx - h)
        ok = (my >= 0) & (my < 2 * h) & (mx >= 0) & (mx < 2 * h)
        mask[my[ok], mx[ok]] = True
        g = self.size // 14
        ring = ndi.binary_dilation(mask, iterations=max(2, int(mask.shape[0] * RING_M / 60 / 2))) & ~mask
        w_in = torch.tensor(np.array(Image.fromarray(mask.astype(np.float32)).resize((g, g), Image.BOX)), device=self.dev)
        w_ring = torch.tensor(np.array(Image.fromarray(ring.astype(np.float32)).resize((g, g), Image.BOX)), device=self.dev)
        win = (cy - h, cy + h, cx - h, cx + h)
        x = torch.tensor(np.stack([self._prep(_crop(A, *win)), self._prep(_crop(B, *win))]), device=self.dev)
        batch = torch.cat([torch.rot90(x, k, (2, 3)) for k in range(4)]).to(self.dtype)  # (8,3,S,S)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.dev.type == "cuda"):
            feats = self._tokens(batch)
        res = []
        for f in feats:                                   # (8, g, g, C)
            acc = 0
            for k in range(4):
                fk = torch.rot90(f[2 * k:2 * k + 2].float(), -k, (1, 2))
                fi = (fk * w_in[None, ..., None]).sum((1, 2)) / (w_in.sum() + 1e-6)
                fr = (fk * w_ring[None, ..., None]).sum((1, 2)) / (w_ring.sum() + 1e-6)
                acc = acc + torch.stack([fi, fr])
            res.append(acc / 4)
        return torch.stack(res).cpu().numpy()


def _l2n(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)


def _cos(u, v):
    return (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-6)


def _featurize(F):
    """F (n, L, 2, 2, C) -> stage-1 inputs (n, 4*L*C), (n, 3*L)"""
    a, b, ra, rb = F[:, :, 0, 0], F[:, :, 0, 1], F[:, :, 1, 0], F[:, :, 1, 1]
    blocks = []
    L = range(F.shape[1])
    for l in L:
        al, bl, ral, rbl = _l2n(a[:, l]), _l2n(b[:, l]), _l2n(ra[:, l]), _l2n(rb[:, l])
        blocks += [al, bl, (bl - al) - (rbl - ral), np.abs(bl - al)]
    scal = ([_cos(a[:, l], b[:, l]) for l in L] + [_cos(ra[:, l], rb[:, l]) for l in L]
            + [_cos(b[:, l], rb[:, l]) - _cos(a[:, l], ra[:, l]) for l in L])
    return np.concatenate(blocks, 1), np.stack(scal, 1)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def score(flt, F, p90):
    """Combined score (logit; > 0 leans real) for blobs with features F and map p90."""
    Xd, Xs = _featurize(F)
    Z = np.concatenate([(Xd - flt["mu"]) @ flt["V"], Xs], 1)
    s1 = ((Z - flt["m1"]) / flt["s1"]) @ flt["th1"][1:] + flt["th1"][0]
    X2 = np.stack([_logit(p90), s1], 1)
    return ((X2 - flt["m2"]) / flt["s2"]) @ flt["th2"][1:] + flt["th2"][0]


def filter_mask(prob, before, after, backbone, gsd, flt, device, low=0.6, redraw=0.83,
                min_score=0.46, progress_fn=None, cancel_fn=None):
    """Binary mask (uint8) of the kept, redrawn blobs, and a stats dict.

    prob: change probabilities (H, W) at the images' resolution; before, after: the
    images (H, W, 3) uint8; gsd: pixel size in metres; low: threshold the blobs are
    found at; redraw: threshold kept blobs are drawn at; min_score: a blob is kept if the
    classifier's probability that it is real is at least this.

    stats: "blobs" (judged), "kept", and for per-polygon scores "labels" (int32 (H, W),
    the blob each pixel belongs to, 0 = none) with "real_prob" (per blob id: the
    classifier's probability that it is real, NaN where not judged).
    """
    import cv2
    from scipy import ndimage as ndi
    min_px = max(1, int(round(MIN_BLOB_M2 / (gsd * gsd))))
    n, comp, st, _ = cv2.connectedComponentsWithStats((prob >= low).astype(np.uint8), connectivity=8)
    ids = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px]
    slices = ndi.find_objects(comp)
    fx = _Features(backbone, device, flt["size"], flt["layers"])
    feats, p90 = [], []
    for j, i in enumerate(ids):
        if cancel_fn and cancel_fn():
            return None, None
        sl = slices[i - 1]
        ys, xs = np.nonzero(comp[sl] == i)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        feats.append(fx.blob(before, after, ys, xs, gsd))
        p90.append(np.percentile(prob[ys, xs], 90))
        if progress_fn:
            progress_fn(j + 1, len(ids))
    out = np.zeros(prob.shape, np.uint8)
    real_prob = np.full(n, np.nan, np.float32)
    if not ids:
        return out, {"blobs": 0, "kept": 0, "labels": comp, "real_prob": real_prob}
    s = score(flt, np.stack(feats), np.array(p90))
    real_prob[ids] = 1 / (1 + np.exp(-s))
    keep = s >= _logit(np.array(min_score))
    for i, k in zip(ids, keep):
        if not k:
            continue
        sl = slices[i - 1]
        m = comp[sl] == i
        q = prob[sl]
        nc, cc, cst, _ = cv2.connectedComponentsWithStats((m & (q >= redraw)).astype(np.uint8), connectivity=8)
        big = np.flatnonzero(cst[1:, cv2.CC_STAT_AREA] >= min_px) + 1
        if big.size:
            reg = np.isin(cc, big)
        else:                                   # no confident part: its own top 20 %
            reg = m & (q >= max(low, 0.8 * q[m].max()))
            if reg.sum() < min_px:
                reg = m
        out[sl][reg] = 1
    return out, {"blobs": len(ids), "kept": int(keep.sum()), "labels": comp, "real_prob": real_prob}
