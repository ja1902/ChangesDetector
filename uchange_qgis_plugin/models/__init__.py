import os
import torch

from .dinov2_cd import DINOv2ChangeDetector, MultiLayerChangeDecoder, VisionTransformer


def build_dinov2_model(checkpoint_path, device):
    # Memory-mapped: the weights are copied into the model straight from the
    # file instead of from a second in-memory copy (saves ~0.35 GB of RAM).
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):     # torch < 2.1, or an old-format file
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt['config']

    backbone = VisionTransformer(
        img_size=cfg['tile_size'],
        patch_size=cfg['patch_size'],
        embed_dim=cfg['embed_dim'],
        depth=cfg['num_blocks'],
        num_heads=cfg['embed_dim'] // 64,
    )
    backbone.load_state_dict(ckpt['backbone'], strict=False)

    decoder = MultiLayerChangeDecoder(
        in_dim=cfg['embed_dim'],
        hidden_dim=cfg['decoder_hidden_dim'],
        out_size=cfg['tile_size'],
        patch_grid=cfg['patch_grid'],
        num_layers=cfg['num_decoder_layers'],
        diff=cfg.get('diff', 'abs'),
        shift_r=cfg.get('shift_r', 1),
    )
    decoder.load_state_dict(ckpt['decoder'], strict=True)

    model = DINOv2ChangeDetector(
        backbone, decoder, cfg['layer_indices'],
    )
    # float16 only on GPU: on CPU it is slow and the float32 inputs would not
    # match the halved weights.
    if device.type == "cuda":
        model.half()
    model.to(device).eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    summary = (f"Model: DINOv2 ViT-B/14 ({params:.1f}M) | "
               f"Checkpoint: {os.path.basename(checkpoint_path)}")
    return model, summary
