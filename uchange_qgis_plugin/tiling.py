import numpy as np
from concurrent.futures import ThreadPoolExecutor


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


def _estimate_batch_size(device, tile_size):
    if device.type != "cuda":
        return 4

    import torch
    free_mem = torch.cuda.mem_get_info(device)[0]

    bytes_per_tile = tile_size * tile_size * 3 * 4
    model_overhead = 1.5 * 1024**3
    activation_multiplier = 12

    available = max(0, free_mem - model_overhead)
    batch = max(1, int(available / (bytes_per_tile * activation_multiplier)))
    return min(batch, 32)


def _prepare_batch(tiles_slice, pre_img, post_img, tile_size):
    from .model_bridge import normalize_tile
    pre_tiles = []
    post_tiles = []
    tile_meta = []
    for y0, x0, y1, x1 in tiles_slice:
        pre_t, th, tw = _extract_tile(pre_img, y0, x0, y1, x1, tile_size)
        post_t, _, _ = _extract_tile(post_img, y0, x0, y1, x1, tile_size)
        pre_tiles.append(normalize_tile(pre_t))
        post_tiles.append(normalize_tile(post_t))
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
        logits = model(pre, post)
        probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()

    return probs


def run_tiled_inference(model, pre_img, post_img, tile_size, overlap, device,
                        progress_fn=None, cancel_fn=None):
    h, w = pre_img.shape[:2]
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    tiles = list(generate_tiles(h, w, tile_size, overlap))
    total = len(tiles)

    batch_size = _estimate_batch_size(device, tile_size)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_prepare_batch, tiles[0:batch_size], pre_img, post_img, tile_size)

        for start in range(0, total, batch_size):
            if cancel_fn and cancel_fn():
                return None

            pre_batch, post_batch, tile_meta = future.result()

            next_start = start + batch_size
            if next_start < total:
                future = executor.submit(
                    _prepare_batch, tiles[next_start:next_start + batch_size],
                    pre_img, post_img, tile_size
                )

            probs = _run_batch(model, pre_batch, post_batch, device)

            for j, (y0, x0, y1, x1, th, tw) in enumerate(tile_meta):
                prob = probs[j, :th, :tw]
                prob_map[y0:y1, x0:x1] += prob
                count_map[y0:y1, x0:x1] += 1.0

            done = min(start + batch_size, total)
            if progress_fn:
                progress_fn(done, total)

    count_map = np.maximum(count_map, 1.0)
    return (prob_map / count_map).astype(np.float32)
