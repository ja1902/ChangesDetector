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

Weaker changes ("orphans", optional, orphan_peak): local peaks of the map (smoothed over
1 m, at least 5 m apart) reaching orphan_peak (0.45) outside every judged blob. Each gets
the part of the map within 15 m that stays above half its peak (at least 50 m2), is judged
by the same classifier and kept at the same min_score, and is drawn as that part. On the
13 squares of our dataset (models trained without the square's city) this found 4 more
changes in 100 at the same false-alarm rate, mostly ones under 100 m2
(work_dirs/exp_orphan/step2.py, benchmarks/paired_eval.py); on pr1_c1 and pr2_c1 (unseen)
it changed nothing at the default min_score and a little at lower ones.

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
BATCH = 4          # blobs per DINOv2 pass in the filter (32 crops)
ORPHAN_SMOOTH_M, ORPHAN_SEP_M, ORPHAN_R_M = 1.0, 5.0, 15.0
_ROWS = 2048              # orphan search works on strips of this many rows

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

    def _inputs(self, A, B, ys, xs, gsd):
        """Crops (2, 3, S, S) and inside / ring weights on the token grid for one blob."""
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
        w_in = np.array(Image.fromarray(mask.astype(np.float32)).resize((g, g), Image.BOX))
        w_ring = np.array(Image.fromarray(ring.astype(np.float32)).resize((g, g), Image.BOX))
        win = (cy - h, cy + h, cx - h, cx + h)
        x = np.stack([self._prep(_crop(A, *win)), self._prep(_crop(B, *win))])
        return x, w_in, w_ring

    @torch.no_grad()
    def blobs(self, A, B, pixels, gsd):
        """[(L, 2 in/ring, 2 dates, 768)] for the blobs given as (ys, xs), averaged over 4
        rotations; BATCH blobs go through DINOv2 together (same result as one at a time)."""
        out = []
        for b0 in range(0, len(pixels), BATCH):
            inp = [self._inputs(A, B, ys, xs, gsd) for ys, xs in pixels[b0:b0 + BATCH]]
            x = torch.tensor(np.stack([i[0] for i in inp]), device=self.dev)       # (k, 2, 3, S, S)
            k, S = x.shape[0], x.shape[-1]
            rot = torch.stack([torch.rot90(x, r, (3, 4)) for r in range(4)], 1)      # (k, 4, 2, 3, S, S)
            batch = rot.reshape(k * 8, 3, S, S).to(self.dtype)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.dev.type == "cuda"):
                feats = self._tokens(batch)
            for j, (_, w_in, w_ring) in enumerate(inp):
                w_in = torch.tensor(w_in, device=self.dev)
                w_ring = torch.tensor(w_ring, device=self.dev)
                res = []
                for f in feats:                                   # (k*8, g, g, C)
                    acc = 0
                    for r in range(4):
                        fk = torch.rot90(f[j * 8 + 2 * r:j * 8 + 2 * r + 2].float(), -r, (1, 2))
                        fi = (fk * w_in[None, ..., None]).sum((1, 2)) / (w_in.sum() + 1e-6)
                        fr = (fk * w_ring[None, ..., None]).sum((1, 2)) / (w_ring.sum() + 1e-6)
                        acc = acc + torch.stack([fi, fr])
                    res.append(acc / 4)
                out.append(torch.stack(res).cpu().numpy())
        return out

    def blob(self, A, B, ys, xs, gsd):
        """(L, 2 in/ring, 2 dates, 768) for one blob, averaged over 4 rotations."""
        return self.blobs(A, B, [(ys, xs)], gsd)[0]


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


def _smoothed(prob, gsd):
    """prob blurred by a 1 m Gaussian (float32), a strip at a time; same as one blur."""
    import cv2
    sig = ORPHAN_SMOOTH_M / gsd
    halo = int(np.ceil(4 * sig)) + 2
    H = prob.shape[0]
    ps = np.empty(prob.shape, np.float32)
    for y0 in range(0, H, _ROWS):
        y1 = min(H, y0 + _ROWS)
        a, b = max(0, y0 - halo), min(H, y1 + halo)
        ps[y0:y1] = cv2.GaussianBlur(prob[a:b].astype(np.float32, copy=False), (0, 0), sig)[y0 - a:y1 - a]
    return ps


def orphans(prob, comp, judged, gsd, peak_level, min_px):
    """Label image (int32) of the orphan candidates and their number.

    comp: blob label image; judged: bool per blob id, True for the blobs the classifier
    judges (orphans stay outside those)."""
    from scipy import ndimage as ndi
    ps = _smoothed(prob, gsd)
    k = 2 * int(round(ORPHAN_SEP_M / gsd)) + 1
    H, W = prob.shape
    pk = np.zeros((H, W), bool)
    for y0 in range(0, H, _ROWS):
        y1 = min(H, y0 + _ROWS)
        a, b = max(0, y0 - k // 2), min(H, y1 + k // 2)
        mx = ndi.maximum_filter(ps[a:b], size=k)[y0 - a:y1 - a]
        q = ps[y0:y1]
        pk[y0:y1] = (q == mx) & (q >= peak_level) & ~judged[comp[y0:y1]]
    lab, n = ndi.label(pk)
    del pk
    centres = np.array(ndi.center_of_mass(lab > 0, lab, range(1, n + 1))).astype(int).reshape(-1, 2) if n else []
    del lab
    out = np.zeros((H, W), np.int32)
    r = int(round(ORPHAN_R_M / gsd))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = yy ** 2 + xx ** 2 <= r * r
    for j, (y, x) in enumerate(centres, 1):
        y0, y1, x0, x1 = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
        d = disk[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)]
        free = ~judged[comp[y0:y1, x0:x1]]
        wl, _ = ndi.label((ps[y0:y1, x0:x1] >= ps[y, x] / 2) & free & d)
        reg = wl == wl[y - y0, x - x0] if wl[y - y0, x - x0] else np.zeros(d.shape, bool)
        reg[y - y0, x - x0] = True
        while reg.sum() < min_px:                              # pad to 50 m2
            reg = ndi.binary_dilation(reg) & free | reg
            if reg.all():
                break
        o = out[y0:y1, x0:x1]
        o[reg & (o == 0)] = j
    del ps
    m = ndi.label(out > 0, structure=np.ones((3, 3)), output=out)
    return out, m


def filter_mask(prob, before, after, backbone, gsd, flt, device, low=0.6, redraw=0.83,
                min_score=0.46, orphan_peak=None, progress_fn=None, cancel_fn=None):
    """Binary mask (uint8) of the kept, redrawn blobs, and a stats dict.

    prob: change probabilities (H, W) at the images' resolution; before, after: the
    images (H, W, 3) uint8; gsd: pixel size in metres; low: threshold the blobs are
    found at; redraw: threshold kept blobs are drawn at; min_score: a blob is kept if the
    classifier's probability that it is real is at least this; orphan_peak: also judge
    the weaker changes peaking at this level outside the blobs (None = off).

    stats: "blobs" (judged), "kept", "orphans" (judged), "orphans_kept", and for
    per-polygon scores "labels" (int32 (H, W), the blob or orphan each pixel belongs to,
    0 = none) with "real_prob" (per id: the classifier's probability that it is real,
    NaN where not judged).
    """
    import cv2
    from scipy import ndimage as ndi
    min_px = max(1, int(round(MIN_BLOB_M2 / (gsd * gsd))))
    n, comp, st, _ = cv2.connectedComponentsWithStats((prob >= low).astype(np.uint8), connectivity=8)
    ids = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px]
    judged = np.zeros(n, bool)
    judged[ids] = True
    olab, m = (orphans(prob, comp, judged, gsd, orphan_peak, min_px) if orphan_peak
               else (None, 0))
    oslices = ndi.find_objects(olab) if m else []
    slices = ndi.find_objects(comp)
    units = [(comp, slices[i - 1], i) for i in ids] + [(olab, sl, j) for j, sl in enumerate(oslices, 1)]
    fx = _Features(backbone, device, flt["size"], flt["layers"])
    feats, p90 = [], []
    for b0 in range(0, len(units), BATCH):
        if cancel_fn and cancel_fn():
            return None, None
        pix = []
        for lab, sl, i in units[b0:b0 + BATCH]:
            ys, xs = np.nonzero(lab[sl] == i)
            pix.append((ys + sl[0].start, xs + sl[1].start))
            p90.append(np.percentile(prob[pix[-1][0], pix[-1][1]], 90))
        feats += fx.blobs(before, after, pix, gsd)
        if progress_fn:
            progress_fn(min(b0 + BATCH, len(units)), len(units))
    out = np.zeros(prob.shape, np.uint8)
    real_prob = np.full(n + m, np.nan, np.float32)
    stats = {"blobs": len(ids), "kept": 0, "orphans": m, "orphans_kept": 0, "labels": comp,
             "real_prob": real_prob}
    if not units:
        return out, stats
    sc = score(flt, np.stack(feats), np.array(p90))
    s, so = sc[:len(ids)], sc[len(ids):]
    real_prob[ids] = 1 / (1 + np.exp(-s))
    cut = _logit(np.array(min_score))
    for j, sl in enumerate(oslices, 1):          # orphans join the labels as ids n .. n+m-1
        reg = olab[sl] == j
        comp[sl][reg] = n + j - 1
        real_prob[n + j - 1] = 1 / (1 + np.exp(-so[j - 1]))
        if so[j - 1] >= cut:
            out[sl][reg] = 1
            stats["orphans_kept"] += 1
    del olab
    keep = s >= cut
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
    stats["kept"] = int(keep.sum())
    return out, stats
