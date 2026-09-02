import os
import sys
import types
import torch

from .changer import ChangerExModel
from .scd_model import SCDModel
from .dinov2_cd import DINOv2ChangeDetector, MultiLayerChangeDecoder, VisionTransformer


def _load_checkpoint(path):
    """Load a checkpoint, tolerating missing mmengine/mmseg imports.

    MMEngine checkpoints pickle HistoryBuffer objects in the 'message_hub'
    key. We only need 'state_dict', so we stub missing modules before loading.
    """
    stubs = {}
    for mod_name in ('mmengine', 'mmengine.logging',
                     'mmengine.logging.history_buffer'):
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            if mod_name == 'mmengine.logging.history_buffer':
                class _Meta(type):
                    def __getattr__(cls, name):
                        return None
                class _HB(metaclass=_Meta):
                    def __init__(self, *a, **k): pass
                    def __setstate__(self, s): pass
                    def __getattr__(self, n): return None
                    def __reduce__(self): return (_HB, ())
                stub.HistoryBuffer = _HB
            sys.modules[mod_name] = stub
            stubs[mod_name] = stub

    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    finally:
        for mod_name in stubs:
            sys.modules.pop(mod_name, None)


def build_changer_model(checkpoint_path, device):
    model = ChangerExModel()
    ckpt = _load_checkpoint(checkpoint_path)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    summary = (f"Model: ChangerEx (R18) | "
               f"Checkpoint: {os.path.basename(checkpoint_path)}")
    return model, summary


def build_scd_model(checkpoint_path, device):
    model = SCDModel()
    ckpt = _load_checkpoint(checkpoint_path)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    summary = (f"Model: SCD UPerNet (R18) | "
               f"Checkpoint: {os.path.basename(checkpoint_path)}")
    return model, summary


def build_dinov2_model(checkpoint_path, device):
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
    )
    decoder.load_state_dict(ckpt['decoder'], strict=True)

    model = DINOv2ChangeDetector(
        backbone, decoder, cfg['layer_indices'],
    )
    model.half().to(device).eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    summary = (f"Model: DINOv2 ViT-B/14 ({params:.1f}M) | "
               f"Checkpoint: {os.path.basename(checkpoint_path)}")
    return model, summary
