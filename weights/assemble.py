"""Build the model weight files from Meta's DINOv2 backbone and the trained parts in this folder.

The DINOv2 change models train only a decoder on top of a frozen DINOv2
ViT-B/14, so this folder holds just the decoders and the land-cover head
(research / non-commercial use). The backbone is Meta's official release
(Apache 2.0), downloaded by the installers from:

    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

Usage (the installers run this):

    python weights/assemble.py OUT_DIR              # exit 3 if the backbone is needed
    python weights/assemble.py OUT_DIR BACKBONE.pth
"""
import hashlib
import os
import shutil
import sys

import torch

BACKBONE_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
BACKBONE_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"

HERE = os.path.dirname(os.path.abspath(__file__))
# full model file -> trained decoder in this folder
MODELS = {
    "dinov2_vitb14_c2s1_levir.pth": "dinov2_vitb14_c2s1_levir.decoder.pth",
    "dinov2_vitb14_levir.pth": "dinov2_vitb14_levir.decoder.pth",
    "dinov2_vitb14_egybcd.pth": "dinov2_vitb14_egybcd.decoder.pth",
}
# used as they are, no backbone needed
HEADS = ["landcover_dinov2_vitb14_oem_second.pth"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    out_dir = argv[1]

    for name in HEADS:
        dest = os.path.join(out_dir, name)
        if not os.path.exists(dest):
            shutil.copyfile(os.path.join(HERE, name), dest)
            print(f"Installed {name}")

    missing = [m for m in MODELS if not os.path.exists(os.path.join(out_dir, m))]
    if not missing:
        print("Model weights already exist, skipping download.")
        return 0
    if len(argv) == 2:
        print("Missing: " + ", ".join(missing))
        return 3

    backbone_path = argv[2]
    if _sha256(backbone_path) != BACKBONE_SHA256:
        print(f"ERROR: {backbone_path} is not Meta's DINOv2 ViT-B/14 file (checksum mismatch).")
        return 1
    backbone = torch.load(backbone_path, map_location="cpu")

    for name in missing:
        part = torch.load(os.path.join(HERE, MODELS[name]), map_location="cpu", weights_only=False)
        dest = os.path.join(out_dir, name)
        tmp = dest + ".part"
        torch.save({"config": part["config"], "backbone": backbone, "decoder": part["decoder"]}, tmp)
        os.replace(tmp, dest)
        print(f"Built {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
