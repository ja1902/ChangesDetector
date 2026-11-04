"""Label-free change score: how unusual each patch's change is, within this image pair.

No training: DINOv2 features of both dates are compared patch by patch (3x3
shift-tolerant cosine distance), and each distance is scored against the typical
change of patches of the same surface type in this pair (k-means clusters per
date). Seasonal fields change a lot everywhere, so they score low; a new building
on bare ground stands out. The score z nudges a trained model's probability:

    p' = sigmoid(logit(p) + w * z)

On held-out squares of the custom dataset (benchmarks/loco_finetune.py, 11 places) it
raised pixel AP of the fine-tuned model from 0.691 to 0.709 and of the shipped
one from 0.614 to 0.655. Port of benchmarks/pair_residual.py and
benchmarks/fuse_residual.py (method "clu16_s", all layers); keep them in step,
the presets' thresholds were chosen on this exact score.
"""
import numpy as np
import torch
import torch.nn.functional as F

PATCH = 14
WINDOW_PATCHES = 37          # DINOv2 window: 518 px, as the models were trained
CLUSTERS = 16
Z_CLIP = (-5.0, 20.0)
BLOCK = 1024                 # source px per block, the size the score was tuned at
OVERLAP = 128

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def _features(model, a, b, scale, device):
    """Multi-layer features of both dates on one patch grid: [(2, C, gh, gw)], (gh, gw), (hs, ws)."""
    h, w = a.shape[:2]
    hs, ws = int(round(h * scale)), int(round(w * scale))
    gh, gw = -(-hs // PATCH), -(-ws // PATCH)
    dtype = next(model.parameters()).dtype
    x = torch.from_numpy(np.stack([a, b])).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(hs, ws), mode="bilinear", align_corners=False) if scale != 1 else x
    x = F.pad(x, (0, gw * PATCH - ws, 0, gh * PATCH - hs), mode="reflect")
    x = ((x - _MEAN) / _STD).to(device).to(dtype)

    def starts(n):
        if n <= WINDOW_PATCHES:
            return [0]
        step = WINDOW_PATCHES - 4
        return sorted(set(list(range(0, n - WINDOW_PATCHES + 1, step)) + [n - WINDOW_PATCHES]))

    wh, ww = min(WINDOW_PATCHES, gh), min(WINDOW_PATCHES, gw)
    acc, cnt = None, torch.zeros(1, 1, gh, gw, device=device)
    for y in starts(gh):
        for xx in starts(gw):
            tile = x[:, :, y * PATCH:(y + wh) * PATCH, xx * PATCH:(xx + ww) * PATCH]
            feats = model.extract_multilayer_features(tile)
            if acc is None:
                acc = [torch.zeros(2, f.shape[-1], gh, gw, device=device) for f in feats]
            for i, f in enumerate(feats):
                acc[i][:, :, y:y + wh, xx:xx + ww] += f.float().transpose(1, 2).reshape(2, -1, wh, ww)
            cnt[:, :, y:y + wh, xx:xx + ww] += 1
    return [f / cnt for f in acc], (gh, gw), (hs, ws)


def _clean_artifacts(f, z_thr=4.0):
    """In-paint DINOv2's high-norm artefact tokens (global information stored in a few
    background patches) with the mean of their non-artefact 3x3 neighbours. In place."""
    n = f.norm(dim=1, keepdim=True)
    flat = n.flatten(1)
    med = flat.median(1).values.view(2, 1, 1, 1)
    mad = (flat - flat.median(1, keepdim=True).values).abs().median(1).values.view(2, 1, 1, 1) * 1.4826 + 1e-6
    bad = ((n - med) / mad > z_thr).float()
    good = 1 - bad
    num = F.avg_pool2d(F.pad(f * good, (1, 1, 1, 1), mode="replicate"), 3, 1)
    den = F.avg_pool2d(F.pad(good, (1, 1, 1, 1), mode="replicate"), 3, 1).clamp(min=1e-6)
    f.copy_(torch.where(bad.bool(), num / den, f))


def _shift_min(target, pred, gh, gw, r=1):
    """min over (2r+1)^2 offsets of ||target_i - pred_{i+d}||."""
    t = target.T.reshape(1, -1, gh, gw)
    p = F.pad(pred.T.reshape(1, -1, gh, gw), (r, r, r, r), mode="replicate")
    best = None
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            d = (t - p[:, :, dy:dy + gh, dx:dx + gw]).pow(2).sum(1).flatten()
            best = d if best is None else torch.minimum(best, d)
    return best.sqrt()


def _kmeans(x, k, iters=12, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        lab = torch.cdist(x, c).argmin(1)
        for j in range(k):
            sel = lab == j
            if sel.any():
                c[j] = x[sel].mean(0)
    return torch.cdist(x, c).argmin(1)


def _cluster_z(r, lab, k):
    z = torch.zeros_like(r)
    for j in range(k):
        sel = lab == j
        if sel.sum() < 8:
            z[sel] = 0.0
            continue
        v = r[sel]
        med = v.median()
        mad = (v - med).abs().median() * 1.4826 + 1e-6
        z[sel] = (v - med) / mad
    return z


def _robust_z(r):
    med = r.median()
    mad = (r - med).abs().median() * 1.4826 + 1e-6
    return (r - med) / mad


def _score_cluster(fa, fb, grid, k=CLUSTERS):
    na, nb = F.normalize(fa, dim=1), F.normalize(fb, dim=1)
    d = 0.5 * (_shift_min(nb, na, *grid) ** 2 + _shift_min(na, nb, *grid) ** 2) / 2
    la = _kmeans(na, k)
    lb = _kmeans(nb, k)
    return 0.5 * (_cluster_z(d, la, k) + _cluster_z(d, lb, k))


def _block_z(model, a, b, scale, device):
    h, w = a.shape[:2]
    feats, (gh, gw), (hs, ws) = _features(model, a, b, scale, device)
    zs = []
    for f in feats:
        _clean_artifacts(f)
        fa = f[0].reshape(f.shape[1], -1).T
        fb = f[1].reshape(f.shape[1], -1).T
        zs.append(_robust_z(_score_cluster(fa, fb, (gh, gw))))
    z = torch.stack(zs).mean(0).reshape(1, 1, gh, gw)
    z = F.interpolate(z, size=(gh * PATCH, gw * PATCH), mode="bilinear", align_corners=False)[:, :, :hs, :ws]
    z = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
    return z[0, 0].clamp(*Z_CLIP).cpu().numpy()


def label_free_z(model, a, b, scale, device, progress_fn=None):
    """The score at the input resolution (float32, h x w), in overlapping blocks.

    model: a DINOv2ChangeDetector (only its backbone is used); scale: the
    inference factor (preset scale x resolution correction).
    """
    h, w = a.shape[:2]
    if max(h, w) <= BLOCK:
        return _block_z(model, a, b, scale, device)

    def starts(n):
        if n <= BLOCK:
            return [0]
        return sorted(set(list(range(0, n - BLOCK + 1, BLOCK - OVERLAP)) + [n - BLOCK]))

    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    blocks = [(y, x) for y in starts(h) for x in starts(w)]
    for i, (y, x) in enumerate(blocks):
        sl = (slice(y, y + BLOCK), slice(x, x + BLOCK))
        acc[sl] += _block_z(model, a[sl], b[sl], scale, device)
        cnt[sl] += 1
        if progress_fn:
            progress_fn(i + 1, len(blocks))
    acc /= np.maximum(cnt, 1)
    return acc


def nudge(prob, z, weight, rows=1024):
    """sigmoid(logit(prob) + weight * z), in place on prob (float32), a strip at a time."""
    for y in range(0, prob.shape[0], rows):
        p = np.clip(prob[y:y + rows], 1e-6, 1 - 1e-6)
        prob[y:y + rows] = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + weight * z[y:y + rows])))
    return prob
