import os
import sys
import types
import torch

from .changer import ChangerExModel
from .scd_model import SCDModel


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
