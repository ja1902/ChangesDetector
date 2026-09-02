import numpy as np
from concurrent.futures import ThreadPoolExecutor


def histogram_match(source, reference):
    """Match the histogram of source to reference, per channel."""
    matched = np.empty_like(source)
    for c in range(source.shape[2]):
        src = source[:, :, c].ravel()
        ref = reference[:, :, c].ravel()
        src_vals, src_idx, src_counts = np.unique(src, return_inverse=True, return_counts=True)
        ref_vals, ref_counts = np.unique(ref, return_counts=True)
        src_cdf = np.cumsum(src_counts).astype(np.float64) / src.size
        ref_cdf = np.cumsum(ref_counts).astype(np.float64) / ref.size
        mapping = np.interp(src_cdf, ref_cdf, ref_vals)
        matched[:, :, c] = mapping[src_idx].reshape(source.shape[:2])
    return matched.astype(source.dtype)


def auto_threshold(prob_map, nbins=256):
    """Find optimal change threshold from the probability map without labels.

    Models the background (unchanged) distribution from the left side of
    its peak in log-probability space and sets the threshold at 4.5 sigma
    above the peak.  This approximates the IoU-optimal operating point
    across different change prevalences and cross-domain shifts.
    """
    eps = 1e-10
    log_probs = np.log(np.clip(prob_map.ravel(), eps, 1.0 - eps))

    hist, bin_edges = np.histogram(log_probs, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if hist.sum() == 0:
        return 0.5

    kernel = np.exp(-np.linspace(-3, 3, 7) ** 2 / 2)
    kernel /= kernel.sum()
    smooth = np.convolve(hist.astype(float), kernel, mode="same")

    n = max(1, len(smooth) - 3)
    left_half = n // 2
    if left_half == 0:
        return 0.5
    peak_idx = int(np.argmax(smooth[:left_half]))
    peak_val = smooth[peak_idx]
    peak_log = bin_centers[peak_idx]

    if peak_val <= 0:
        return 0.5

    half_max = peak_val / 2.0
    hwhm_idx = 0
    for i in range(peak_idx - 1, -1, -1):
        if smooth[i] < half_max:
            hwhm_idx = i
            break
    hwhm_left = peak_log - bin_centers[hwhm_idx]
    if hwhm_left <= 0:
        return 0.5

    sigma = hwhm_left / 1.177
    threshold = float(np.exp(peak_log + 4.5 * sigma))
    return min(threshold, 0.5)


def generate_tiles(height, width, tile_size, overlap):
    """Yield (y_start, x_start, y_end, x_end) for each tile."""
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("Overlap must be smaller than tile_size")

    y_positions = list(range(0, height, stride))
    x_positions = list(range(0, width, stride))

    for y0 in y_positions:
        for x0 in x_positions:
            y_end = min(y0 + tile_size, height)
            x_end = min(x0 + tile_size, width)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            yield y_start, x_start, y_end, x_end


def _extract_tile(img, y0, x0, y1, x1, tile_size):
    tile = img[y0:y1, x0:x1]
    th, tw = tile.shape[:2]
    if th < tile_size or tw < tile_size:
        padded = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
        padded[:th, :tw] = tile
        return padded, th, tw
    return tile, th, tw


def _estimate_batch_size(device, tile_size, scd_mode=False, model=None):
    if device.type != "cuda":
        return 4

    import torch
    free_mem = torch.cuda.mem_get_info(device.index or 0)[0]

    is_vit = model is not None and hasattr(model, 'extract_multilayer_features')

    if is_vit:
        patch_size = getattr(model.backbone.patch_embed, 'patch_size', 14)
        seq_len = (tile_size // patch_size) ** 2
        depth = len(list(model.backbone.blocks))
        num_heads = 12
        if hasattr(model.backbone.blocks[0], 'attn'):
            num_heads = model.backbone.blocks[0].attn.num_heads
        # Attention scores dominate: num_heads * seq_len^2 * 2 bytes (fp16) per layer
        # Plus FFN, residuals, decoder features — use 4x multiplier on attention
        bytes_per_tile = num_heads * seq_len * seq_len * 2 * depth * 4
        model_overhead = 512 * 1024**2
    else:
        bytes_per_tile = tile_size * tile_size * 3 * 4
        model_overhead = (2.5 if scd_mode else 1.5) * 1024**3
        activation_multiplier = 16 if scd_mode else 12
        bytes_per_tile *= activation_multiplier

    available = max(0, free_mem - model_overhead)
    batch = max(1, int(available / bytes_per_tile))
    return min(batch, 32)


def _prepare_batch(tiles_slice, pre_img, post_img, tile_size, grayscale=False,
                    hist_match=False):
    from .model_bridge import normalize_tile
    pre_tiles = []
    post_tiles = []
    tile_meta = []
    for y0, x0, y1, x1 in tiles_slice:
        pre_t, th, tw = _extract_tile(pre_img, y0, x0, y1, x1, tile_size)
        post_t, _, _ = _extract_tile(post_img, y0, x0, y1, x1, tile_size)
        if hist_match:
            post_t = histogram_match(post_t, pre_t)
        pre_tiles.append(normalize_tile(pre_t, grayscale=grayscale))
        post_tiles.append(normalize_tile(post_t, grayscale=grayscale))
        tile_meta.append((y0, x0, y1, x1, th, tw))
    pre_batch = np.stack(pre_tiles)
    post_batch = np.stack(post_tiles)
    return pre_batch, post_batch, tile_meta


def _get_amp_dtype(device):
    import torch
    if device.type != "cuda":
        return None
    # Ampere (sm_80+) has native bfloat16; older GPUs (Turing, Volta) use float16
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _run_batch(model, pre_batch, post_batch, device):
    import torch

    pre = torch.from_numpy(pre_batch).float()
    post = torch.from_numpy(post_batch).float()

    if device.type == "cuda":
        pre = pre.pin_memory().to(device, non_blocking=True)
        post = post.pin_memory().to(device, non_blocking=True)
    else:
        pre = pre.to(device)
        post = post.to(device)

    amp_dtype = _get_amp_dtype(device)
    use_amp = amp_dtype is not None

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype or torch.float32):
        output = model(pre, post)

        if isinstance(output, dict):
            binary_probs = torch.softmax(output['seg_logits'].float(), dim=1)[:, 1].cpu().numpy()
            sem_from = output['seg_logits_from'].float().argmax(dim=1).cpu().numpy()
            sem_to = output['seg_logits_to'].float().argmax(dim=1).cpu().numpy()
            return {'binary_probs': binary_probs, 'semantic_from': sem_from, 'semantic_to': sem_to}

        probs = torch.softmax(output.float(), dim=1)[:, 1].cpu().numpy()

    return probs


def run_tiled_inference(model, pre_img, post_img, tile_size, overlap, device,
                        progress_fn=None, cancel_fn=None, grayscale=False,
                        hist_match=False):
    h, w = pre_img.shape[:2]
    scd_mode = getattr(model, 'is_scd', False)

    tiles = list(generate_tiles(h, w, tile_size, overlap))
    total = len(tiles)

    batch_size = _estimate_batch_size(device, tile_size, scd_mode=scd_mode, model=model)

    while True:
        prob_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        if scd_mode:
            num_classes = 6
            votes_from = np.zeros((num_classes, h, w), dtype=np.float32)
            votes_to = np.zeros((num_classes, h, w), dtype=np.float32)

        oom = False

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_prepare_batch, tiles[0:batch_size], pre_img, post_img, tile_size, grayscale, hist_match)

            for start in range(0, total, batch_size):
                if cancel_fn and cancel_fn():
                    return None

                pre_batch, post_batch, tile_meta = future.result()

                next_start = start + batch_size
                if next_start < total:
                    future = executor.submit(
                        _prepare_batch, tiles[next_start:next_start + batch_size],
                        pre_img, post_img, tile_size, grayscale, hist_match
                    )

                try:
                    result = _run_batch(model, pre_batch, post_batch, device)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower() or batch_size <= 1:
                        raise
                    import torch
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    oom = True
                    break

                if scd_mode:
                    for j, (y0, x0, y1, x1, th, tw) in enumerate(tile_meta):
                        prob_map[y0:y1, x0:x1] += result['binary_probs'][j, :th, :tw]
                        count_map[y0:y1, x0:x1] += 1.0
                        cls_from = result['semantic_from'][j, :th, :tw]
                        cls_to = result['semantic_to'][j, :th, :tw]
                        for c in range(num_classes):
                            votes_from[c, y0:y1, x0:x1] += (cls_from == c).astype(np.float32)
                            votes_to[c, y0:y1, x0:x1] += (cls_to == c).astype(np.float32)
                else:
                    for j, (y0, x0, y1, x1, th, tw) in enumerate(tile_meta):
                        prob_map[y0:y1, x0:x1] += result[j, :th, :tw]
                        count_map[y0:y1, x0:x1] += 1.0

                done = min(start + batch_size, total)
                if progress_fn:
                    progress_fn(done, total)

        if not oom:
            break

    count_map = np.maximum(count_map, 1.0)
    prob_map = (prob_map / count_map).astype(np.float32)

    if scd_mode:
        semantic_from = votes_from.argmax(axis=0).astype(np.uint8)
        semantic_to = votes_to.argmax(axis=0).astype(np.uint8)
        return {
            'prob_map': prob_map,
            'semantic_from': semantic_from,
            'semantic_to': semantic_to,
        }

    return prob_map
