"""False-alarm filter for change polygons (finetuned model).

The change map is thresholded low, and each blob gets a second opinion: a small
classifier trained on our real / false verdicts looks at frozen DINOv2 features of the
blob and a ring around it, on both dates, together with the map's own confidence in the
blob. Blobs judged false (mostly building lean and shadows) are dropped. Each kept blob
is then redrawn as its part above the usual threshold, so outlines stay as tight as
without the filter.

Tested on the 8 held-out squares of our dataset (each scored by a model and classifier
trained without its place) and on pr2_c1, which neither had seen (EXPERIMENTS.md): at the
same share of changes
found, 20-45 % fewer false alarms; on pr2 16/23 found with 14 false alarms per km2,
against 28 without the filter. It can drop a real change that touches a big new shadow,
because the low threshold merges the two into one blob.

Weaker changes ("orphans", optional, orphan_peak): local peaks of the map (smoothed over
1 m, at least 5 m apart) reaching orphan_peak (0.45) outside every judged blob. Each gets
the part of the map within 15 m that stays above half its peak (at least 50 m2), is judged
by the same classifier and kept at the same min_score, and is drawn as that part. On the
13 squares of our dataset (models trained without the square's city) this found 4 more
changes in 100 at the same false-alarm rate, mostly ones under 100 m2 (EXPERIMENTS.md);
on pr1_c1 and pr2_c1 (unseen)
it changed nothing at the default min_score and a little at lower ones.

Only valid on the map it was trained on: the finetuned model at x1.4 with flip
averaging and WITHOUT the label-free nudge (preset "synthetic" pipeline). The backbone is
frozen and the same in every DINOv2 checkpoint here, so the change model's backbone is
reused. Weights: weights/falsealarm_filter_finetuned.npz.
"""
import numpy as np
import torch
from PIL import Image

MIN_BLOB_M2 = 50.0        # smaller blobs were never judged (nor scored)
CONTEXT_M, MIN_HALF_M = 15.0, 20.0
RING_M = 8.0
BATCH = 4          # blobs per DINOv2 pass in the filter (32 crops)
_CROPS = 8 * BATCH        # crops per DINOv2 pass (2 dates per blob and rotation)
# Cascade: every candidate is first judged on one rotation of its crops; the other three
# are added only where that quick score comes within CASCADE_MARGIN (logit) of the keep
# cut. On 15 squares (3,772 candidates) the quick score was never more than 1.21 below the
# full one with either filter (99.9% within 1.01), so a candidate it rules out is far from
# being kept; those that are finished get exactly the full score (bit for bit). At the
# default cut this skips ~40% of the filter's work, less at lower cuts.
CASCADE_MARGIN = 1.75
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


def backbone_tokens(bb, x, size, layers):
    """Patch tokens (n, g, g, C) of the crops x (n, 3, size, size) at the given blocks."""
    g = size // 14
    x = bb.patch_embed(x)
    pe = bb._interpolate_pos_embed(x.shape[1])
    x = x + pe[:, 1:]
    x = torch.cat([(bb.cls_token + pe[:, :1]).expand(x.shape[0], -1, -1), x], 1)
    out = []
    for i, blk in enumerate(bb.blocks):
        x = blk(x)
        if i in layers:
            out.append(x[:, 1:].reshape(x.shape[0], g, g, -1))
        if i >= max(layers):
            break
    return out


class _Features:
    def __init__(self, backbone, device, size, layers, tokens_fn=None):
        self.bb, self.dev, self.size, self.layers = backbone, device, int(size), tuple(int(l) for l in layers)
        self.dtype = next(backbone.parameters()).dtype
        self.max_items = _CROPS // 2          # (blob, rotation) items per DINOv2 pass
        self.tokens_fn = tokens_fn            # the same pass in another runtime (openvino_cpu)

    @torch.no_grad()
    def _tokens(self, x):
        if self.tokens_fn is not None:
            return self.tokens_fn(x)
        return backbone_tokens(self.bb, x, self.size, self.layers)

    def _tokens_fitting(self, xs):
        """_tokens of the crops xs (k, 2, 3, S, S), in smaller passes if the GPU runs out
        of memory (the features of a crop do not depend on the others in its pass)."""
        k, S = xs.shape[0], xs.shape[-1]
        while True:
            step = min(k, self.max_items)
            parts = []
            try:
                for i in range(0, k, step):
                    batch = torch.tensor(xs[i:i + step], device=self.dev).reshape(-1, 3, S, S).to(self.dtype)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.dev.type == "cuda"):
                        parts.append(self._tokens(batch))
                return parts[0] if len(parts) == 1 else [torch.cat(f) for f in zip(*parts)]
            except RuntimeError as e:             # torch.OutOfMemoryError is one
                if "out of memory" not in str(e).lower() or step <= 1:
                    raise
            parts = batch = None                  # outside the handler: frees the tensors
            torch.cuda.empty_cache()
            self.max_items = max(1, step // 2)

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

    def _rotated(self, A, B, pixels, gsd, items):
        """Crops (len(items), 2, 3, S, S) float32 for (blob, rotation) items, and the
        token-grid weights {blob: (w_in, w_ring)} of their blobs."""
        xs, weights, last = [], {}, (None, None)
        for u, r in items:
            if last[0] != u:
                x, w_in, w_ring = self._inputs(A, B, *pixels[u], gsd)
                weights[u] = (w_in, w_ring)
                last = (u, x)
            xs.append(np.rot90(last[1], r, (2, 3)))
        return np.stack(xs), weights

    @torch.no_grad()
    def rotations(self, A, B, pixels, gsd, items, progress_fn=None, cancel_fn=None):
        """{(blob, rotation): (L, 2 in/ring, 2 dates, 768)} for the (blob index, rotation
        0-3) items; the crops of _CROPS // 2 items go through DINOv2 together, while the
        next ones are prepared on the CPU. None if cancelled."""
        from concurrent.futures import ThreadPoolExecutor
        step = _CROPS // 2
        chunks = [items[i:i + step] for i in range(0, len(items), step)]
        out = {}
        with ThreadPoolExecutor(max_workers=1) as pool:
            nxt = pool.submit(self._rotated, A, B, pixels, gsd, chunks[0]) if chunks else None
            for c, chunk in enumerate(chunks):
                if cancel_fn and cancel_fn():
                    nxt.cancel()
                    return None
                xs, weights = nxt.result()
                if c + 1 < len(chunks):
                    nxt = pool.submit(self._rotated, A, B, pixels, gsd, chunks[c + 1])
                feats = self._tokens_fitting(xs)
                res = []
                for j, (u, r) in enumerate(chunk):
                    w_in, w_ring = weights[u]
                    w_in = torch.tensor(w_in, device=self.dev)
                    w_ring = torch.tensor(w_ring, device=self.dev)
                    per_layer = []
                    for f in feats:                               # (2k, g, g, C)
                        fk = torch.rot90(f[2 * j:2 * j + 2].float(), -r, (1, 2))
                        fi = (fk * w_in[None, ..., None]).sum((1, 2)) / (w_in.sum() + 1e-6)
                        fr = (fk * w_ring[None, ..., None]).sum((1, 2)) / (w_ring.sum() + 1e-6)
                        per_layer.append(torch.stack([fi, fr]))
                    res.append(torch.stack(per_layer))
                for item, v in zip(chunk, torch.stack(res).cpu().numpy()):
                    out[item] = v
                if progress_fn:
                    progress_fn(c + 1, len(chunks))
        return out

    @staticmethod
    def averaged(rot):
        """Mean over the 4 rotations, summed in the order the filter was trained with."""
        acc = 0
        for r in range(4):
            acc = acc + rot[r]
        return acc / 4

    def blobs(self, A, B, pixels, gsd):
        """[(L, 2 in/ring, 2 dates, 768)] for the blobs given as (ys, xs), averaged over 4
        rotations (crops of several blobs go through DINOv2 together: same result as one
        at a time)."""
        rot = self.rotations(A, B, pixels, gsd, [(u, r) for u in range(len(pixels)) for r in range(4)])
        return [self.averaged([rot[(u, r)] for r in range(4)]) for u in range(len(pixels))]

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


class _Pixels:
    """(ys, xs) of each judged unit (label image, slice, id), found when asked for."""

    def __init__(self, units):
        self.units = units

    def __len__(self):
        return len(self.units)

    def __getitem__(self, k):
        lab, sl, i = self.units[k]
        ys, xs = np.nonzero(lab[sl] == i)
        return ys + sl[0].start, xs + sl[1].start


def filter_mask(prob, before, after, backbone, gsd, flt, device, low=0.6, redraw=0.83,
                min_score=0.46, orphan_peak=None, progress_fn=None, cancel_fn=None,
                cascade_margin=CASCADE_MARGIN, tokens_fn=None):
    """Binary mask (uint8) of the kept, redrawn blobs, and a stats dict.

    prob: change probabilities (H, W) at the images' resolution; before, after: the
    images (H, W, 3) uint8; gsd: pixel size in metres; low: threshold the blobs are
    found at; redraw: threshold kept blobs are drawn at; min_score: a blob is kept if the
    classifier's probability that it is real is at least this; orphan_peak: also judge
    the weaker changes peaking at this level outside the blobs (None = off);
    cascade_margin: judge each candidate on one rotation first and finish only those whose
    quick score is within this of the cut (None = all four rotations for every one);
    tokens_fn: runs the backbone's pass instead of PyTorch (openvino_cpu.filter_tokens).

    stats: "blobs" (judged), "kept", "orphans" (judged), "orphans_kept", and for
    per-polygon scores "labels" (int32 (H, W), the blob or orphan each pixel belongs to,
    0 = none) with "real_prob" (per id: the classifier's probability that it is real,
    NaN where not judged; from the quick 1-rotation look for candidates it ruled out),
    "finished" (candidates judged on all four rotations).
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
    pix = _Pixels(units)
    p90 = [np.percentile(prob[ys, xs], 90) for ys, xs in pix]
    cut = _logit(np.array(min_score))
    fx = _Features(backbone, device, flt["size"], flt["layers"], tokens_fn)
    nu = len(units)
    cascade = cascade_margin is not None
    # progress in (blob, rotation) items; until the quick look is done, assume every
    # candidate needs finishing, so the bar never moves back
    work = [nu * (1 if cascade else 4), 3 * nu if cascade else 0]

    def report(offset, n_items):
        if progress_fn:
            return lambda c, t: progress_fn(offset + min(c * (_CROPS // 2), n_items), sum(work))

    first = [(u, r) for u in range(nu) for r in ((0,) if cascade else range(4))]
    rot = fx.rotations(before, after, pix, gsd, first, report(0, len(first)), cancel_fn)
    if rot is None:
        return None, None
    finish = []
    if cascade and nu:
        quick = score(flt, np.stack([rot[(u, 0)] for u in range(nu)]), np.array(p90))
        finish = [u for u in range(nu) if quick[u] >= cut - cascade_margin]
        work[1] = 3 * len(finish)
        more = fx.rotations(before, after, pix, gsd, [(u, r) for u in finish for r in (1, 2, 3)],
                            report(work[0], work[1]), cancel_fn)
        if more is None:
            return None, None
        rot.update(more)
    full = set(finish) if cascade else set(range(nu))
    # A candidate the quick look rules out keeps its 1-rotation features (and score).
    feats = [fx.averaged([rot[(u, r)] for r in range(4)]) if u in full else rot[(u, 0)]
             for u in range(nu)]
    out = np.zeros(prob.shape, np.uint8)
    real_prob = np.full(n + m, np.nan, np.float32)
    stats = {"blobs": len(ids), "kept": 0, "orphans": m, "orphans_kept": 0, "labels": comp,
             "real_prob": real_prob, "finished": len(full)}
    if not units:
        return out, stats
    sc = score(flt, np.stack(feats), np.array(p90))
    s, so = sc[:len(ids)], sc[len(ids):]
    real_prob[ids] = 1 / (1 + np.exp(-s))
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
