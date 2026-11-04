#!/usr/bin/env python3
"""Label-free change from frozen DINOv2 features: learn each pair's nuisance, keep the residual.

Most of a scene is unchanged, so a small model fitted on the pair itself, mapping
the "before" features to the "after" features, learns that pair's season / sensor /
illumination differences. It is too small to learn rare events, so construction and
demolition stay in its residual. No training data: everything is fitted per pair.

Scored on the fixed benchmark pairs (benchmarks/sets/) with bench.py's metrics:
best IoU over all thresholds and AP, pooled over the set. Compared with:
  cos        1 - cosine similarity of the raw features (the old zero-shot baseline)
  std        per-image channel standardisation, then L2 (a diagonal nuisance model)
  ridge      robust ridge regression before -> after (and after -> before) in a
             per-pair PCA space, residual measured with the inlier residual covariance
Each also in a shift-tolerant form (suffix _s): the residual at a patch is the best
match over its 3x3 neighbours, for misregistration and off-nadir lean.

    python benchmarks/pair_residual.py                         # s2looking, ViT-B/14
    python benchmarks/pair_residual.py --sets s2looking,levir,egybcd --k 16,32,64
    python benchmarks/pair_residual.py --small --save-png out/  # quick look, writes maps
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import uchange_qgis_plugin.model_bridge  # noqa: E402,F401  (puts the venv on sys.path)
from bench import Scores, data_root, load_set, pair_paths, read_rgb, append_result, git_state  # noqa: E402

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PATCH = 14


# ── Features ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract(model, a, b, scale, win_patches, device):
    """Multi-layer DINOv2 features of both dates on one patch grid.

    Returns a list (one per layer) of (2, C, gh, gw) float32 tensors, and (gh, gw).
    Images bigger than a window are run in overlapping windows whose features are
    averaged, as the decoder sees them in the plugin.
    """
    h, w = a.shape[:2]
    hs, ws = int(round(h * scale)), int(round(w * scale))
    gh, gw = -(-hs // PATCH), -(-ws // PATCH)
    x = torch.from_numpy(np.stack([a, b])).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(hs, ws), mode="bilinear", align_corners=False) if scale != 1 else x
    x = F.pad(x, (0, gw * PATCH - ws, 0, gh * PATCH - hs), mode="reflect")
    x = ((x - _MEAN) / _STD).to(device).half()

    def starts(n):
        if n <= win_patches:
            return [0]
        step = win_patches - 4
        return sorted(set(list(range(0, n - win_patches + 1, step)) + [n - win_patches]))

    wh, ww = min(win_patches, gh), min(win_patches, gw)
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


# ── Change scores (all on an (N, C) pair of feature matrices) ────────────────

def shift_min(target, pred, gh, gw, r=1):
    """min over (2r+1)^2 offsets of ||target_i - pred_{i+d}||, both (N, k) in a metric space."""
    t = target.T.reshape(1, -1, gh, gw)
    p = F.pad(pred.T.reshape(1, -1, gh, gw), (r, r, r, r), mode="replicate")
    best = None
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            d = (t - p[:, :, dy:dy + gh, dx:dx + gw]).pow(2).sum(1).flatten()
            best = d if best is None else torch.minimum(best, d)
    return best.sqrt()


def score_cos(fa, fb, grid, shift):
    na, nb = F.normalize(fa, dim=1), F.normalize(fb, dim=1)
    if not shift:
        return 1 - (na * nb).sum(1)
    # 1 - cos = ||na - nb||^2 / 2 for unit vectors
    return 0.5 * (shift_min(nb, na, *grid) ** 2 + shift_min(na, nb, *grid) ** 2) / 2


def score_std(fa, fb, grid, shift):
    za = (fa - fa.mean(0)) / (fa.std(0) + 1e-6)
    zb = (fb - fb.mean(0)) / (fb.std(0) + 1e-6)
    if not shift:
        return (za - zb).norm(dim=1)
    return 0.5 * (shift_min(zb, za, *grid) + shift_min(za, zb, *grid))


def robust_fit(src, dst, alpha, keep, iters):
    """Weighted ridge dst ~ [src, 1] W, refitted on the `keep` share of best-fitting rows.

    Returns predictions for all rows and a whitening matrix for the residual metric
    (inverse Cholesky of the inlier residual covariance).
    """
    n, k = src.shape
    xs = torch.cat([src, torch.ones(n, 1, device=src.device)], 1)
    # alpha relative to the mean diagonal of src^T src
    reg = alpha * (src.pow(2).sum() / k) * torch.eye(k + 1, device=src.device)
    reg[-1, -1] = 0
    w = torch.ones(n, 1, device=src.device)
    for _ in range(iters + 1):
        wt = torch.linalg.solve(xs.T @ (w * xs) + reg, xs.T @ (w * dst))
        pred = xs @ wt
        err = dst - pred
        inl = err[w[:, 0] > 0]
        cov = inl.T @ inl / max(len(inl) - 1, 1)
        cov = cov + 1e-4 * cov.diagonal().mean() * torch.eye(k, device=src.device)
        white = torch.linalg.inv(torch.linalg.cholesky(cov)).T      # x @ white: whitened
        r = (err @ white).norm(dim=1)
        w = (r <= torch.quantile(r, keep)).float()[:, None]
    return pred, white


def score_ridge(fa, fb, grid, shift, k=32, alpha=1e-2, keep=0.85, iters=3, whiten=True):
    a = fa - fa.mean(0)
    b = fb - fb.mean(0)
    cov = (a.T @ a + b.T @ b) / (2 * len(a))
    _, vecs = torch.linalg.eigh(cov)
    p = vecs[:, -k:]
    a, b = a @ p, b @ p
    out = 0
    for src, dst in ((a, b), (b, a)):
        pred, white = robust_fit(src, dst, alpha, keep, iters)
        if not whiten:
            white = torch.eye(white.shape[0], device=white.device)
        if shift:
            out = out + shift_min(dst @ white, pred @ white, *grid)
        else:
            out = out + ((dst - pred) @ white).norm(dim=1)
    return out / 2


def clean_artifacts(f, z_thr=4.0):
    """In-paint DINOv2's high-norm artefact tokens. f: (2, C, gh, gw), cleaned in place.

    DINOv2 without registers stores global information in a few background patches,
    whose features are unrelated to the image content there. They are found by
    feature norm and replaced by the mean of their non-artefact 3x3 neighbours.
    """
    n = f.norm(dim=1, keepdim=True)                                   # (2, 1, gh, gw)
    flat = n.flatten(1)
    med = flat.median(1).values.view(2, 1, 1, 1)
    mad = (flat - flat.median(1, keepdim=True).values).abs().median(1).values.view(2, 1, 1, 1) * 1.4826 + 1e-6
    bad = ((n - med) / mad > z_thr).float()
    good = 1 - bad
    num = F.avg_pool2d(F.pad(f * good, (1, 1, 1, 1), mode="replicate"), 3, 1)
    den = F.avg_pool2d(F.pad(good, (1, 1, 1, 1), mode="replicate"), 3, 1).clamp(min=1e-6)
    f.copy_(torch.where(bad.bool(), num / den, f))
    return bad.mean().item()


def kmeans(x, k, iters=12, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        lab = torch.cdist(x, c).argmin(1)
        for j in range(k):
            sel = lab == j
            if sel.any():
                c[j] = x[sel].mean(0)
    return torch.cdist(x, c).argmin(1)


def cluster_z(r, lab, k):
    """Robust z of r within each cluster: how unusual a change is for that kind of surface."""
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


def score_cluster(fa, fb, grid, shift, k=16):
    """cos(_s) distance, normalised by the typical change of the patch's surface type.

    Surfaces are clustered separately on each date; a patch scores high when its
    change is unusual for the surface it had before (A clusters) and for the one it
    has after (B clusters). Seasonal fields change a lot everywhere and are scored
    against other fields; bare ground or roofs rarely change, so a change there stands out.
    """
    d = score_cos(fa, fb, grid, shift)
    la = kmeans(F.normalize(fa, dim=1), k)
    lb = kmeans(F.normalize(fb, dim=1), k)
    return 0.5 * (cluster_z(d, la, k) + cluster_z(d, lb, k))


def robust_z(r):
    med = r.median()
    mad = (r - med).abs().median() * 1.4826 + 1e-6
    return (r - med) / mad


# ── Main ─────────────────────────────────────────────────────────────────────

def methods(ks):
    m = {"cos": lambda fa, fb, g: score_cos(fa, fb, g, False),
         "cos_s": lambda fa, fb, g: score_cos(fa, fb, g, True),
         "std": lambda fa, fb, g: score_std(fa, fb, g, False),
         "std_s": lambda fa, fb, g: score_std(fa, fb, g, True)}
    for c in (8, 16, 32):
        m[f"clu{c}_s"] = lambda fa, fb, g, c=c: score_cluster(fa, fb, g, True, k=c)
    m["clu16"] = lambda fa, fb, g: score_cluster(fa, fb, g, False, k=16)
    for k in ks:
        m[f"ridge{k}"] = lambda fa, fb, g, k=k: score_ridge(fa, fb, g, False, k=k)
        m[f"ridge{k}_s"] = lambda fa, fb, g, k=k: score_ridge(fa, fb, g, True, k=k)
        m[f"ridge{k}e"] = lambda fa, fb, g, k=k: score_ridge(fa, fb, g, False, k=k, whiten=False)
        m[f"ridge{k}e_s"] = lambda fa, fb, g, k=k: score_ridge(fa, fb, g, True, k=k, whiten=False)
    return m


def to_prob(z, gh, gw, hs, ws, h, w, zmax):
    """Patch-grid score -> (h, w) map in [0, 1] (monotone, for the threshold sweep)."""
    m = z.reshape(1, 1, gh, gw)
    m = F.interpolate(m, size=(gh * PATCH, gw * PATCH), mode="bilinear", align_corners=False)
    m = m[:, :, :hs, :ws]
    if (hs, ws) != (h, w):
        m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
    return (m[0, 0] / zmax).clamp(0, 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sets", default="s2looking")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--weights", default="dinov2_vitb14_c2s1_levir.pth",
                    help="checkpoint whose (frozen, original DINOv2) backbone is used")
    ap.add_argument("--scale", type=float, default=1.4)
    ap.add_argument("--min-side", type=int, default=518,
                    help="upsample small images (LEVIR / EGY-BCD tiles) to at least this side, "
                         "so a pair has enough patches to fit on")
    ap.add_argument("--win", type=int, default=37, help="window in patches (37 = 518 px)")
    ap.add_argument("--k", default="32", help="PCA dims for ridge, comma list")
    ap.add_argument("--only", default=None, help="comma list of methods to run")
    ap.add_argument("--save-png", default=None, help="folder for side-by-side maps of the first pairs")
    ap.add_argument("--n-png", type=int, default=8)
    ap.add_argument("--name", default="pair-residual")
    ap.add_argument("--no-clean", action="store_true", help="keep DINOv2's high-norm artefact tokens")
    args = ap.parse_args()

    from uchange_qgis_plugin.model_bridge import build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(str(ROOT / args.weights), device, model_type="dinov2")
    model.eval()
    root = data_root(args.data)
    ms = methods([int(k) for k in args.k.split(",")])
    if args.only:
        ms = {k: v for k, v in ms.items() if k in args.only.split(",")}
    layer_sets = {"all": [0, 1, 2, 3], "L2": [0], "L5": [1], "L8": [2], "L11": [3], "L8+11": [2, 3]}

    row = {"kind": "pair_residual", "name": args.name, "weights": args.weights, "scale": args.scale,
           "small": args.small, "time": time.strftime("%Y-%m-%d %H:%M"), "device": device.type,
           **git_state(), "sets": {}}
    for set_name in args.sets.split(","):
        stems = load_set(set_name, args.small)
        maps = {m: [] for m in ms}          # per pair: (4, N) robust z per layer
        geo, labels, art = [], [], []
        t0 = time.time()
        for i, stem in enumerate(stems):
            pa, pb, pl = pair_paths(root, set_name, stem)
            a, b = read_rgb(pa), read_rgb(pb)
            labels.append(np.array(Image.open(pl).convert("L")) > 127)
            h, w = a.shape[:2]
            scale = max(args.scale, args.min_side / min(h, w))
            feats, (gh, gw), (hs, ws) = extract(model, a, b, scale, args.win, device)
            geo.append((gh, gw, hs, ws, h, w))
            if not args.no_clean:
                frac = [clean_artifacts(f) for f in feats]
                art.append(frac)
            for m, fn in ms.items():
                per_layer = []
                for f in feats:
                    fa = f[0].reshape(f.shape[1], -1).T
                    fb = f[1].reshape(f.shape[1], -1).T
                    per_layer.append(robust_z(fn(fa, fb, (gh, gw))))
                maps[m].append(torch.stack(per_layer))
            print(f"\r{set_name}: {i + 1}/{len(stems)}  ({time.time() - t0:.0f}s)", end="", flush=True)
        print()
        if art:
            print("artefact tokens in-painted per layer (%):", " ".join(f"{100 * v:.2f}" for v in np.mean(art, 0)))

        res = {}
        for m in ms:
            for ls_name, ls in layer_sets.items():
                zs = [mp[ls].mean(0) for mp in maps[m]]
                zmax = torch.quantile(torch.cat(zs)[::7], 0.999).item() if sum(z.numel() for z in zs) > 1 else 1.0
                sc = Scores()
                for z, g, lab in zip(zs, geo, labels):
                    prob = to_prob(z, *g, zmax)
                    sc.add(prob, prob >= 0.5, lab)
                s = sc.summary(False)
                res[f"{m}/{ls_name}"] = {"best_iou": s["best_iou"], "ap": s["ap"],
                                         "best_threshold": s["best_threshold"]}
        row["sets"][set_name] = res
        print(f"\n{set_name} ({len(stems)} pairs)   best IoU    AP      [layers]")
        best = {}
        for key, s in res.items():
            m, ls_name = key.split("/")
            if m not in best or s["best_iou"] > res[best[m]]["best_iou"]:
                best[m] = key
        for m, key in best.items():
            s, a_ = res[key], res[f"{m}/all"]
            print(f"  {m:12s}  all layers {a_['best_iou']:.3f}  {a_['ap']:.3f}   "
                  f"best: {s['best_iou']:.3f}  {s['ap']:.3f}  [{key.split('/')[1]}]")

        if args.save_png:
            save_pngs(args, set_name, stems, root, maps, geo, labels)
    append_result(row)


def save_pngs(args, set_name, stems, root, maps, geo, labels):
    """before | after | label | cos | best ridge, for the first pairs."""
    out = Path(args.save_png) / set_name
    out.mkdir(parents=True, exist_ok=True)
    show = [m for m in ("cos", "cos_s", "clu16_s") if m in maps] + [m for m in maps if m.startswith("ridge")][-1:]
    for i, stem in enumerate(stems[:args.n_png]):
        pa, pb, _ = pair_paths(root, set_name, stem)
        a, b = read_rgb(pa), read_rgb(pb)
        tiles = [a, b, np.repeat(labels[i][..., None] * 255, 3, 2).astype(np.uint8)]
        for m in show:
            z = maps[m][i].mean(0)
            p = to_prob(z, *geo[i], torch.quantile(z, 0.995).item() + 1e-6)
            tiles.append(np.repeat((p * 255).astype(np.uint8)[..., None], 3, 2))
        side = 384
        tiles = [np.array(Image.fromarray(t).resize((side, side))) for t in tiles]
        Image.fromarray(np.concatenate(tiles, 1)).save(out / f"{stem}.jpg", quality=88)
    (out / "columns.txt").write_text("before | after | label | " + " | ".join(show) + "\n")


if __name__ == "__main__":
    main()
