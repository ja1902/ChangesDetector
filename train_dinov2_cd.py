"""
Standalone DINOv2 change detection training script.
Siamese frozen DINOv2 backbone + lightweight trainable decoder.
Trained on LEVIR-CD+, tested on S2Looking.
"""
import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import v2
from PIL import Image


# ── Dataset ──────────────────────────────────────────────────────────────────

class CDDataset(Dataset):
    """Bitemporal change detection dataset. Works with LEVIR-CD+ and S2Looking."""

    def __init__(self, root, split="train", crop_size=518, augment=True,
                 dir_a="A", dir_b="B", dir_label="label",
                 fda_target_dir=None, fda_beta=0.05):
        self.root = Path(root) / split
        self.crop_size = crop_size
        self.augment = augment

        self.dir_a = self.root / dir_a
        self.dir_b = self.root / dir_b
        self.dir_label = self.root / dir_label

        self.filenames = sorted([
            f.name for f in self.dir_a.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        ])
        self.fda_beta = fda_beta
        self.fda_target_images = []
        if fda_target_dir:
            target_path = Path(fda_target_dir)
            for subdir in ["Image1", "Image2"]:
                d = target_path / subdir
                if d.exists():
                    self.fda_target_images.extend([
                        str(f) for f in d.iterdir()
                        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                    ])
            print(f"  [FDA] Loaded {len(self.fda_target_images)} target domain images")
        print(f"  [{split}] Found {len(self.filenames)} image pairs in {self.root}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_a = Image.open(self.dir_a / fname).convert("RGB")
        img_b = Image.open(self.dir_b / fname).convert("RGB")
        label = Image.open(self.dir_label / fname).convert("L")

        img_a = np.array(img_a)
        img_b = np.array(img_b)
        label = np.array(label)

        # Binarize label (handle 0/255 or 0/1)
        label = (label > 127).astype(np.uint8)

        if self.augment:
            img_a, img_b, label = self._augment(img_a, img_b, label)
        else:
            img_a, img_b, label = self._resize(img_a, img_b, label)

        # To tensor, float, normalize (ImageNet stats)
        img_a = self._to_tensor(img_a)
        img_b = self._to_tensor(img_b)
        label = torch.from_numpy(label).long()

        return img_a, img_b, label

    def _augment(self, a, b, label):
        h, w = a.shape[:2]
        cs = self.crop_size

        # Random crop
        if h > cs or w > cs:
            top = np.random.randint(0, max(h - cs, 1))
            left = np.random.randint(0, max(w - cs, 1))
            a = a[top:top+cs, left:left+cs]
            b = b[top:top+cs, left:left+cs]
            label = label[top:top+cs, left:left+cs]

        # Resize to exact crop_size if needed
        if a.shape[0] != cs or a.shape[1] != cs:
            a = np.array(Image.fromarray(a).resize((cs, cs), Image.BILINEAR))
            b = np.array(Image.fromarray(b).resize((cs, cs), Image.BILINEAR))
            label = np.array(Image.fromarray(label).resize((cs, cs), Image.NEAREST))

        # Random horizontal flip
        if np.random.rand() > 0.5:
            a = np.flip(a, axis=1).copy()
            b = np.flip(b, axis=1).copy()
            label = np.flip(label, axis=1).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            a = np.flip(a, axis=0).copy()
            b = np.flip(b, axis=0).copy()
            label = np.flip(label, axis=0).copy()

        # Random temporal swap
        if np.random.rand() > 0.5:
            a, b = b, a

        # FDA: swap low-frequency spectrum with target domain
        if self.fda_target_images and np.random.rand() > 0.5:
            target_img = np.array(Image.open(
                self.fda_target_images[np.random.randint(len(self.fda_target_images))]
            ).convert("RGB").resize((cs, cs), Image.BILINEAR))
            a = self._fda_transfer(a, target_img, self.fda_beta)
            b = self._fda_transfer(b, target_img, self.fda_beta)

        return a, b, label

    @staticmethod
    def _fda_transfer(source, target, beta):
        """Fourier Domain Adaptation: swap low-frequency amplitude of source with target."""
        src_f = np.fft.fft2(source.astype(np.float32), axes=(0, 1))
        tgt_f = np.fft.fft2(target.astype(np.float32), axes=(0, 1))
        src_amp = np.abs(src_f)
        src_pha = np.angle(src_f)
        tgt_amp = np.abs(tgt_f)
        h, w = source.shape[:2]
        bh, bw = int(h * beta), int(w * beta)
        # Build low-frequency mask (center of shifted spectrum)
        mask = np.zeros((h, w, 1), dtype=np.float32)
        mask[:bh, :bw] = 1
        mask[:bh, w-bw:] = 1
        mask[h-bh:, :bw] = 1
        mask[h-bh:, w-bw:] = 1
        # Swap low-freq amplitude
        new_amp = src_amp * (1 - mask) + tgt_amp * mask
        result_f = new_amp * np.exp(1j * src_pha)
        result = np.real(np.fft.ifft2(result_f, axes=(0, 1)))
        return np.clip(result, 0, 255).astype(np.uint8)

    def _resize(self, a, b, label):
        cs = self.crop_size
        a = np.array(Image.fromarray(a).resize((cs, cs), Image.BILINEAR))
        b = np.array(Image.fromarray(b).resize((cs, cs), Image.BILINEAR))
        label = np.array(Image.fromarray(label).resize((cs, cs), Image.NEAREST))
        return a, b, label

    def _to_tensor(self, img):
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(t)
        return t


# ── Model ────────────────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch, out_ch, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ── BAN (Bitemporal Adapter Network) components ─────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


class SideEncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=2, downsample=True):
        super().__init__()
        layers = []
        if downsample:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GELU())
            in_ch = out_ch
        for _ in range(num_blocks):
            layers.append(DepthwiseSeparableConv(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class LightweightSideEncoder(nn.Module):
    """4-stage convolutional encoder producing multi-scale features.
    Replaces MixVisionTransformer (MIT-b0) from BAN paper."""

    CHANNELS = [32, 64, 160, 256]

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.stages = nn.ModuleList([
            SideEncoderStage(32, 32, num_blocks=2, downsample=False),
            SideEncoderStage(32, 64, num_blocks=2, downsample=True),
            SideEncoderStage(64, 160, num_blocks=2, downsample=True),
            SideEncoderStage(160, 256, num_blocks=2, downsample=True),
        ])

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs


class BridgeLayer(nn.Module):
    """Cross-attention bridge: side encoder features attend to frozen backbone features.
    Adapted from BAN paper's BridgeLayer."""

    def __init__(self, side_dim, backbone_dim, num_heads=4):
        super().__init__()
        self.proj_backbone = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, side_dim, bias=False),
        )
        self.norm_side = nn.LayerNorm(side_dim)
        self.cross_attn = nn.MultiheadAttention(side_dim, num_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(side_dim)
        self.ffn = nn.Sequential(
            nn.Linear(side_dim, side_dim * 4),
            nn.GELU(),
            nn.Linear(side_dim * 4, side_dim),
        )
        self.scale = nn.Parameter(torch.ones(side_dim) * 1e-5)

    def forward(self, side_feat, backbone_feat, patch_grid):
        B, C, H, W = side_feat.shape
        side_flat = side_feat.flatten(2).permute(0, 2, 1)
        bb_proj = self.proj_backbone(backbone_feat)
        q = self.norm_side(side_flat)
        attn_out, _ = self.cross_attn(q, bb_proj, bb_proj)
        side_flat = side_flat + attn_out * self.scale
        x = self.norm_ffn(side_flat)
        side_flat = side_flat + self.ffn(x) * self.scale
        out = side_flat.permute(0, 2, 1).reshape(B, C, H, W)
        bb_spatial = bb_proj.permute(0, 2, 1).reshape(B, -1, patch_grid, patch_grid)
        bb_resized = F.interpolate(bb_spatial, size=(H, W), mode="bilinear", align_corners=False)
        return out + bb_resized


class BANChangeDecoder(nn.Module):
    """BAN-style change decoder: side encoder + bridge layers + MLP head.
    Processes both timestamps through shared side encoder with backbone feature injection."""

    def __init__(self, backbone_dim=768, patch_grid=37, out_size=518,
                 fusion_stages=(1, 2, 3)):
        super().__init__()
        side_channels = LightweightSideEncoder.CHANNELS
        self.side_encoder = LightweightSideEncoder()
        self.fusion_stages = fusion_stages
        self.patch_grid = patch_grid
        self.out_size = out_size

        self.bridges = nn.ModuleDict()
        heads_map = {32: 2, 64: 4, 160: 5, 256: 8}
        for stage_idx in fusion_stages:
            ch = side_channels[stage_idx]
            self.bridges[str(stage_idx)] = BridgeLayer(
                side_dim=ch, backbone_dim=backbone_dim,
                num_heads=heads_map[ch])

        fused_ch = sum(side_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.discriminator = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 2, 1),
        )

    def _encode_one(self, img, backbone_feats):
        x = self.side_encoder.stem(img)
        outs = []
        for i, stage in enumerate(self.side_encoder.stages):
            x = stage(x)
            if i in self.fusion_stages:
                x = self.bridges[str(i)](x, backbone_feats[i], self.patch_grid)
            outs.append(x)
        target_size = outs[0].shape[2:]
        resized = []
        for feat in outs:
            resized.append(F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False))
        fused = self.fusion(torch.cat(resized, dim=1))
        return fused

    def forward(self, feats_a, feats_b, img_a=None, img_b=None):
        enc_a = self._encode_one(img_a, feats_a)
        enc_b = self._encode_one(img_b, feats_b)
        x = torch.cat([enc_a, enc_b], dim=1)
        x = x + self.discriminator(x)
        x = self.head(x)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x


class MultiLayerChangeDecoder(nn.Module):
    """Multi-layer decoder using features from 4 ViT layers.
    Each layer's features are projected, then progressively upsampled and fused
    (FPN-style) to produce a high-resolution change prediction."""

    def __init__(self, in_dim=768, hidden_dim=256, out_size=518, patch_grid=37,
                 num_layers=4):
        super().__init__()
        self.patch_grid = patch_grid
        self.out_size = out_size
        self.num_layers = num_layers

        # Per-layer: normalize features then project [A, B, |A-B|] from 3*in_dim to hidden_dim
        self.layer_norms = nn.ModuleList([nn.LayerNorm(in_dim) for _ in range(num_layers)])
        self.layer_projs = nn.ModuleList([
            _conv_bn_relu(in_dim * 3, hidden_dim, kernel=1, padding=0)
            for _ in range(num_layers)
        ])

        # Progressive upsampling stages (each 2x upsample + fuse + refine)
        # Stage 0: patch_grid (37x37) → 2x (74x74)
        # Stage 1: 74x74 → 2x (148x148)
        # Stage 2: 148x148 → 2x (296x296)
        self.upsample_blocks = nn.ModuleList()
        for i in range(3):
            in_ch = hidden_dim * 2 if i > 0 else hidden_dim
            # After concat with skip from next layer: hidden_dim + hidden_dim
            fuse_in = hidden_dim + hidden_dim
            self.upsample_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ))

        # Fusion convs after skip connection concat
        self.fuse_convs = nn.ModuleList([
            nn.Sequential(
                _conv_bn_relu(hidden_dim * 2, hidden_dim),
                _conv_bn_relu(hidden_dim, hidden_dim),
            )
            for _ in range(3)
        ])

        # Final refinement and prediction at ~296x296
        self.head = nn.Sequential(
            _conv_bn_relu(hidden_dim, hidden_dim // 2),
            _conv_bn_relu(hidden_dim // 2, hidden_dim // 4),
            nn.Conv2d(hidden_dim // 4, 2, 1),
        )

    def forward(self, feats_a, feats_b, out_size=None):
        """
        feats_a, feats_b: list of 4 tensors, each (B, N, D) from different ViT layers.
        out_size: optional (H, W) tuple for output resolution. Defaults to self.out_size.
        """
        B, N, _ = feats_a[0].shape
        H = W = int(N ** 0.5)

        # Normalize and project each layer's difference features
        projected = []
        for i in range(self.num_layers):
            fa_normed = self.layer_norms[i](feats_a[i])
            fb_normed = self.layer_norms[i](feats_b[i])
            fa = fa_normed.permute(0, 2, 1).reshape(B, -1, H, W)
            fb = fb_normed.permute(0, 2, 1).reshape(B, -1, H, W)
            diff = torch.abs(fa - fb)
            x = torch.cat([fa, fb, diff], dim=1)
            projected.append(self.layer_projs[i](x))

        # Progressive upsampling with skip connections
        x = projected[3]
        for i in range(3):
            skip = projected[2 - i]
            x = self.upsample_blocks[i](x)
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.fuse_convs[i](x)

        x = self.head(x)
        target = out_size if out_size is not None else self.out_size
        if isinstance(target, int):
            target = (target, target)
        x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        return x


class SimpleChangeDecoder(nn.Module):
    """Simple decoder for single-layer features (e.g. AnySat).
    Takes single feature grid per timestamp, computes difference, upsamples."""

    def __init__(self, in_dim=768, hidden_dim=256, out_size=500, patch_grid=50):
        super().__init__()
        self.patch_grid = patch_grid
        self.out_size = out_size

        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            _conv_bn_relu(hidden_dim * 3, hidden_dim),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            _conv_bn_relu(hidden_dim, hidden_dim),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 2, stride=2),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            _conv_bn_relu(hidden_dim // 2, hidden_dim // 4),
            nn.Conv2d(hidden_dim // 4, 2, 1),
        )

    def forward(self, feats_a, feats_b):
        B = feats_a.shape[0]
        H = W = self.patch_grid
        fa = self.proj(feats_a).permute(0, 2, 1).reshape(B, -1, H, W)
        fb = self.proj(feats_b).permute(0, 2, 1).reshape(B, -1, H, W)
        diff = torch.abs(fa - fb)
        x = torch.cat([fa, fb, diff], dim=1)
        x = self.spatial(x)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x


class DINOv2ChangeDetector(nn.Module):
    """Siamese DINO backbone + decoder for change detection.
    Supports FPN and BAN decoder types, DINOv2 and DINOv3 backbones."""

    def __init__(self, backbone, decoder, layer_indices, family="dinov2",
                 decoder_type="fpn"):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.layer_indices = layer_indices
        self.family = family
        self.decoder_type = decoder_type
        self._hook_features = {}

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        if family == "dinov3":
            for idx in layer_indices:
                self.backbone.blocks[idx].register_forward_hook(
                    self._make_hook(f"block_{idx}")
                )
            self.n_prefix = self.backbone.num_prefix_tokens

    def _make_hook(self, name):
        def hook(module, input, output):
            self._hook_features[name] = output
        return hook

    def extract_multilayer_features(self, x):
        if self.family == "dinov3":
            self._hook_features.clear()
            self.backbone(x)
            feats = []
            for idx in self.layer_indices:
                f = self._hook_features[f"block_{idx}"]
                feats.append(f[:, self.n_prefix:, :])
            return feats
        else:
            return list(self.backbone.get_intermediate_layers(
                x, n=self.layer_indices, reshape=False
            ))

    def forward(self, img_a, img_b):
        with torch.no_grad():
            feats_a = self.extract_multilayer_features(img_a)
            feats_b = self.extract_multilayer_features(img_b)
        if self.decoder_type == "ban":
            logits = self.decoder(feats_a, feats_b, img_a=img_a, img_b=img_b)
        else:
            logits = self.decoder(feats_a, feats_b)
        return logits

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


class LoRAReins(nn.Module):
    """LoRA-Reins adapter from CrossEarth (IEEE TPAMI 2025).
    Injects learnable link tokens at each transformer layer via low-rank decomposition
    to make DINOv2 features more domain-invariant."""

    def __init__(self, embed_dim=1024, num_layers=24, token_length=100, lora_dim=16):
        super().__init__()
        self.num_layers = num_layers
        self.token_length = token_length
        self.embed_dim = embed_dim
        self.learnable_tokens_a = nn.Parameter(torch.randn(num_layers, token_length, lora_dim) * 0.02)
        self.learnable_tokens_b = nn.Parameter(torch.randn(num_layers, lora_dim, embed_dim) * 0.02)
        self.mlp_token2feat = nn.Linear(embed_dim, embed_dim)
        self.mlp_delta_f = nn.Linear(embed_dim, embed_dim)
        self.scale = nn.Parameter(torch.tensor(1e-3))
        self.merge = nn.Linear(embed_dim * 3 // 4, embed_dim // 4)
        self.transform = nn.Linear(embed_dim, embed_dim // 4)

    def get_tokens(self, layer_idx):
        return self.learnable_tokens_a[layer_idx] @ self.learnable_tokens_b[layer_idx]

    def forward(self, patch_features, link_token_output):
        token_feat = self.mlp_token2feat(link_token_output.mean(dim=1))
        delta = self.mlp_delta_f(token_feat)
        return patch_features + self.scale * delta.unsqueeze(1)


class CrossEarthChangeDetector(nn.Module):
    """CrossEarth DINOv2 ViT-L/16 + LoRA-Reins + FPN decoder for change detection.
    Both backbone and Reins are frozen; only the decoder is trained."""

    def __init__(self, backbone, reins, decoder, layer_indices):
        super().__init__()
        self.backbone = backbone
        self.reins = reins
        self.decoder = decoder
        self.layer_indices = layer_indices

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        for p in self.reins.parameters():
            p.requires_grad = False
        self.reins.eval()

    def extract_multilayer_features(self, x):
        B = x.shape[0]
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.norm_pre(x)

        multi_layer_feats = []
        for i, block in enumerate(self.backbone.blocks):
            link_tokens = self.reins.get_tokens(i).unsqueeze(0).expand(B, -1, -1)
            x_aug = torch.cat([x, link_tokens], dim=1)
            x_aug = block(x_aug)
            x = x_aug[:, :-self.reins.token_length]
            reins_out = x_aug[:, -self.reins.token_length:]
            patches = x[:, 1:]
            patches = self.reins(patches, reins_out)
            x = torch.cat([x[:, :1], patches], dim=1)

            if i in self.layer_indices:
                multi_layer_feats.append(patches)

        return multi_layer_feats

    def forward(self, img_a, img_b):
        with torch.no_grad():
            feats_a = self.extract_multilayer_features(img_a)
            feats_b = self.extract_multilayer_features(img_b)
        logits = self.decoder(feats_a, feats_b)
        return logits

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.reins.eval()
        return self


class ResNetFPNChangeDecoder(nn.Module):
    """FPN decoder for ResNet multi-scale features. Handles different channel dims
    at each scale: layer1(256), layer2(512), layer3(1024), layer4(2048) for R50."""

    def __init__(self, channels=(256, 512, 1024, 2048), hidden_dim=256, out_size=518):
        super().__init__()
        self.out_size = out_size
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 3, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ) for ch in channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                _conv_bn_relu(hidden_dim, hidden_dim),
                _conv_bn_relu(hidden_dim, hidden_dim),
            ) for _ in range(len(channels) - 1)
        ])
        self.head = nn.Sequential(
            _conv_bn_relu(hidden_dim, hidden_dim // 2),
            _conv_bn_relu(hidden_dim // 2, hidden_dim // 4),
            nn.Conv2d(hidden_dim // 4, 2, 1),
        )

    def forward(self, feats_a, feats_b):
        laterals = []
        for i in range(len(feats_a)):
            diff = torch.abs(feats_a[i] - feats_b[i])
            x = torch.cat([feats_a[i], feats_b[i], diff], dim=1)
            laterals.append(self.lateral_convs[i](x))

        x = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            x = F.interpolate(x, size=laterals[i].shape[2:], mode="bilinear", align_corners=False)
            x = x + laterals[i]
            x = self.fpn_convs[i](x)

        x = self.head(x)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x


class CACoChangeDetector(nn.Module):
    """CACo ResNet-50 backbone + ResNet FPN decoder for change detection."""

    def __init__(self, backbone, decoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def extract_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)
        return [c1, c2, c3, c4]

    def forward(self, img_a, img_b):
        with torch.no_grad():
            feats_a = self.extract_features(img_a)
            feats_b = self.extract_features(img_b)
        logits = self.decoder(feats_a, feats_b)
        return logits

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


class DOFAChangeDetector(nn.Module):
    """DOFA (Dynamic One-For-All) backbone + FPN decoder for change detection.
    Uses wavelength-conditioned patch embedding for sensor-agnostic features."""

    def __init__(self, backbone, decoder, layer_indices, wavelengths):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.layer_indices = layer_indices
        self.wavelengths = wavelengths

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def extract_multilayer_features(self, x):
        wavelist = torch.tensor(self.wavelengths, device=x.device).float()
        patches, _ = self.backbone.patch_embed(x, wavelist)
        patches = patches + self.backbone.pos_embed[:, 1:, :]
        cls_token = self.backbone.cls_token + self.backbone.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(patches.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, patches), dim=1)

        feats = []
        for idx, block in enumerate(self.backbone.blocks):
            tokens = block(tokens)
            if idx in self.layer_indices:
                feats.append(tokens[:, 1:])
        return feats

    def forward(self, img_a, img_b):
        with torch.no_grad():
            feats_a = self.extract_multilayer_features(img_a)
            feats_b = self.extract_multilayer_features(img_b)
        logits = self.decoder(feats_a, feats_b)
        return logits

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


class AnySatChangeDetector(nn.Module):
    """Siamese AnySat backbone + decoder for change detection.
    Uses AnySat 'spot' modality (3ch RGB, 1m) for feature extraction."""

    def __init__(self, backbone, decoder, crop_size=500):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.crop_size = crop_size

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def extract_features(self, x):
        data = {"spot": x}
        feat = self.backbone(data, patch_size=10, output='patch')
        B, H, W, C = feat.shape
        return feat.reshape(B, H * W, C)

    def forward(self, img_a, img_b):
        with torch.no_grad():
            feats_a = self.extract_features(img_a)
            feats_b = self.extract_features(img_b)
        logits = self.decoder(feats_a, feats_b)
        return logits

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


# ── Metrics ──────────────────────────────────────────────────────────────────

class IoUMeter:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)

    def update(self, pred, target):
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        for c in range(self.num_classes):
            p = (pred == c)
            t = (target == c)
            self.intersection[c] += (p & t).sum()
            self.union[c] += (p | t).sum()

    def get_iou(self):
        iou = self.intersection / (self.union + 1e-10)
        return iou

    def get_miou(self):
        return self.get_iou().mean()


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    meter = IoUMeter()
    t0 = time.time()

    for i, (img_a, img_b, label) in enumerate(loader):
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(img_a, img_b)
            loss = F.cross_entropy(logits, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        meter.update(pred, label)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            iou = meter.get_iou()
            print(f"  [Epoch {epoch}] iter {i+1}/{len(loader)}  "
                  f"loss={total_loss/(i+1):.4f}  "
                  f"Unchanged_IoU={iou[0]:.4f}  Changed_IoU={iou[1]:.4f}  "
                  f"mIoU={meter.get_miou():.4f}  "
                  f"({elapsed:.0f}s)")

    avg_loss = total_loss / len(loader)
    iou = meter.get_iou()
    print(f"  [Epoch {epoch}] TRAIN  loss={avg_loss:.4f}  "
          f"Unchanged_IoU={iou[0]:.4f}  Changed_IoU={iou[1]:.4f}  "
          f"mIoU={meter.get_miou():.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device, split="val"):
    model.eval()
    meter = IoUMeter()
    total_loss = 0

    for img_a, img_b, label in loader:
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        label = label.to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(img_a, img_b)
            loss = F.cross_entropy(logits, label)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        meter.update(pred, label)

    avg_loss = total_loss / len(loader)
    iou = meter.get_iou()
    miou = meter.get_miou()
    print(f"  [{split.upper()}]  loss={avg_loss:.4f}  "
          f"Unchanged_IoU={iou[0]:.4f}  Changed_IoU={iou[1]:.4f}  "
          f"mIoU={miou:.4f}")
    return miou, iou


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DINOv2 Change Detection")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--backbone", choices=["vits14", "vitb14", "vitl14", "dinov3_vitb16", "dinov3_vitl16_sat", "anysat", "crossearth", "dofa_vitb16", "caco_r50"], default="vitb14")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to backbone weights (.pth or .safetensors). If omitted, uses timm pretrained.")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to training dataset (e.g. LEVIR-CD)")
    parser.add_argument("--test-data-root", type=str, default=None,
                        help="Path to test dataset (e.g. S2Looking). If omitted, uses val split of data-root.")
    parser.add_argument("--test-dir-a", type=str, default="Image1")
    parser.add_argument("--test-dir-b", type=str, default="Image2")
    parser.add_argument("--output-dir", type=str, default="work_dirs/dinov2_cd")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=518,
                        help="Must be multiple of 14 for DINOv2")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to decoder checkpoint to resume from")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--decoder", choices=["fpn", "ban"], default="fpn",
                        help="Decoder type: fpn (multi-layer FPN) or ban (Bitemporal Adapter Network)")
    parser.add_argument("--fda-target", type=str, default=None,
                        help="Path to target domain images for FDA (e.g. S2Looking/test)")
    parser.add_argument("--fda-beta", type=float, default=0.05,
                        help="FDA beta: fraction of low-frequency spectrum to swap")
    parser.add_argument("--extra-data", type=str, nargs="+", default=[],
                        help="Additional data roots to concatenate for training")
    parser.add_argument("--all-splits", action="store_true",
                        help="Use train+val+test from primary data-root (for cross-dataset eval)")
    args = parser.parse_args()

    BACKBONE_CFG = {
        "vits14":           {"timm": "vit_small_patch14_dinov2",      "dim": 384,  "blocks": 12, "patch": 14, "family": "dinov2"},
        "vitb14":           {"timm": "vit_base_patch14_dinov2",       "dim": 768,  "blocks": 12, "patch": 14, "family": "dinov2"},
        "vitl14":           {"timm": "vit_large_patch14_dinov2",      "dim": 1024, "blocks": 24, "patch": 14, "family": "dinov2"},
        "dinov3_vitb16":    {"timm": "vit_base_patch16_dinov3",       "dim": 768,  "blocks": 12, "patch": 16, "family": "dinov3"},
        "dinov3_vitl16_sat":{"timm": "vit_large_patch16_dinov3",      "dim": 1024, "blocks": 24, "patch": 16, "family": "dinov3",
                             "pretrained_tag": "vit_large_patch16_dinov3.sat493m"},
        "anysat":           {"dim": 768, "family": "anysat"},
        "crossearth":       {"dim": 1024, "blocks": 24, "patch": 16, "family": "crossearth"},
        "dofa_vitb16":      {"dim": 768,  "blocks": 12, "patch": 16, "family": "dofa"},
        "caco_r50":         {"dim": 2048, "family": "caco"},
    }
    cfg = BACKBONE_CFG[args.backbone]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    embed_dim = cfg["dim"]

    if cfg["family"] == "anysat":
        # AnySat: crop_size must produce integer patch grid at 1m/10m patch
        # 500 pixels at 1m → 500m / 10m patch = 50 patches
        if args.crop_size != 500:
            print(f"  [AnySat] Overriding crop_size from {args.crop_size} to 500 (must be multiple of 10 for spot modality)")
            args.crop_size = 500
        patch_grid = args.crop_size // 10  # 50x50

        print(f"Loading AnySat ...")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anysat_repo", "src"))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "anysat_repo"))
        from hubconf import AnySat
        backbone = AnySat(model_size='base', flash_attn=False)
        if args.weights:
            state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            backbone.model.load_state_dict(state_dict)
        else:
            raise ValueError("AnySat requires --weights path to anysat_base.pth")

        print(f"  Backbone loaded: embed_dim={embed_dim}, patch_grid={patch_grid}x{patch_grid}")

        decoder = SimpleChangeDecoder(
            in_dim=embed_dim, hidden_dim=256,
            out_size=args.crop_size, patch_grid=patch_grid,
        )
        model = AnySatChangeDetector(backbone, decoder, crop_size=args.crop_size).to(device)
    elif cfg["family"] == "dofa":
        if args.crop_size != 224:
            print(f"  [DOFA] Overriding crop_size from {args.crop_size} to 224 (DOFA default)")
            args.crop_size = 224
        patch_size = cfg["patch"]
        patch_grid = args.crop_size // patch_size  # 14x14
        num_blocks = cfg["blocks"]
        layer_indices = [2, 5, 8, 11]

        print(f"Loading DOFA ViT-B/16 ...")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dofa_repo"))
        from dofa_v1 import vit_base_patch16
        backbone = vit_base_patch16(num_classes=0, global_pool=False)
        dofa_weights = args.weights or "weights/DOFA_ViT_base_e100.pth"
        sd = torch.load(dofa_weights, map_location="cpu", weights_only=False)
        backbone.load_state_dict(sd, strict=False)
        print(f"  Backbone loaded: embed_dim={embed_dim}, patch_grid={patch_grid}x{patch_grid}")

        decoder = MultiLayerChangeDecoder(
            in_dim=embed_dim, hidden_dim=256,
            out_size=args.crop_size, patch_grid=patch_grid, num_layers=4,
        )
        wavelengths = [0.665, 0.56, 0.49]
        model = DOFAChangeDetector(backbone, decoder, layer_indices, wavelengths).to(device)
    elif cfg["family"] == "caco":
        print(f"Loading CACo ResNet-50 ...")
        from torchvision.models import resnet50
        backbone = resnet50(weights=None)
        caco_weights = args.weights or "weights/caco_resnet50_1m.pth"
        proxy = nn.Sequential(*list(resnet50(weights=None).children())[:-1], nn.Flatten())
        proxy.load_state_dict(torch.load(caco_weights, map_location="cpu", weights_only=False))
        for i, (cp, cb) in enumerate(zip(list(proxy.children()), list(backbone.children()))):
            if i >= 8: break
            for pp, pb in zip(cp.parameters(), cb.parameters()):
                pb.data.copy_(pp.data)
        del proxy
        print(f"  Backbone loaded: CACo ResNet-50 pretrained on 1M Sentinel-2 locations")

        decoder = ResNetFPNChangeDecoder(
            channels=(256, 512, 1024, 2048), hidden_dim=256, out_size=args.crop_size,
        )
        model = CACoChangeDetector(backbone, decoder).to(device)
    elif cfg["family"] == "crossearth":
        if args.crop_size != 512:
            print(f"  [CrossEarth] Overriding crop_size from {args.crop_size} to 512 (ViT-L/16 expects 512)")
            args.crop_size = 512
        patch_size = cfg["patch"]
        patch_grid = args.crop_size // patch_size  # 32x32
        num_blocks = cfg["blocks"]
        layer_indices = [5, 11, 17, 23]

        print(f"Loading CrossEarth DINOv2 ViT-L/16 + LoRA-Reins ...")
        from timm.models.vision_transformer import VisionTransformer
        backbone = VisionTransformer(
            img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
            mlp_ratio=4, init_values=1e-5, num_classes=0, global_pool='',
        )
        bb_weights = args.weights or "weights/crossearth_dinov2.pth"
        bb_sd = torch.load(bb_weights, map_location="cpu", weights_only=False)
        backbone.load_state_dict(bb_sd, strict=False)
        print(f"  Backbone loaded: embed_dim={embed_dim}, patch_grid={patch_grid}x{patch_grid}")

        reins = LoRAReins(embed_dim=1024, num_layers=24, token_length=100, lora_dim=16)
        reins_path = "weights/crossearth_reins.pth"
        reins_sd = torch.load(reins_path, map_location="cpu", weights_only=False)["state_dict"]
        reins_dict = {k.replace("backbone.reins.", ""): v for k, v in reins_sd.items() if "reins" in k}
        reins.load_state_dict(reins_dict)
        print(f"  Reins loaded: {sum(p.numel() for p in reins.parameters())/1e6:.2f}M params")

        decoder = MultiLayerChangeDecoder(
            in_dim=embed_dim, hidden_dim=256,
            out_size=args.crop_size, patch_grid=patch_grid, num_layers=4,
        )
        model = CrossEarthChangeDetector(backbone, reins, decoder, layer_indices).to(device)
    else:
        patch_size = cfg["patch"]
        assert args.crop_size % patch_size == 0, f"crop_size must be multiple of {patch_size}, got {args.crop_size}"
        patch_grid = args.crop_size // patch_size
        num_blocks = cfg["blocks"]
        layer_indices = [num_blocks // 4 - 1, num_blocks // 2 - 1, 3 * num_blocks // 4 - 1, num_blocks - 1]

        print(f"Loading {args.backbone} ...")
        import timm

        if cfg["family"] == "dinov3":
            backbone = timm.create_model(cfg["timm"], pretrained=False, num_classes=0)
            if args.weights:
                if args.weights.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(args.weights)
                else:
                    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
                backbone.load_state_dict(state_dict, strict=False)
            else:
                timm_tag = cfg.get("pretrained_tag", cfg["timm"])
                backbone = timm.create_model(timm_tag, pretrained=True, num_classes=0)
        else:
            backbone = timm.create_model(cfg["timm"], pretrained=False, num_classes=0)
            if args.weights:
                state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
                backbone.load_state_dict(state_dict, strict=False)
            else:
                raise ValueError("DINOv2 backbones require --weights")

        print(f"  Backbone loaded: embed_dim={embed_dim}, patch_grid={patch_grid}x{patch_grid}")
        print(f"  Using layers: {layer_indices} (0-indexed, out of {num_blocks})")

        if args.decoder == "ban":
            decoder = BANChangeDecoder(
                backbone_dim=embed_dim,
                patch_grid=patch_grid,
                out_size=args.crop_size,
                fusion_stages=(1, 2, 3),
            )
        else:
            decoder = MultiLayerChangeDecoder(
                in_dim=embed_dim,
                hidden_dim=256,
                out_size=args.crop_size,
                patch_grid=patch_grid,
                num_layers=4,
            )
        model = DINOv2ChangeDetector(
            backbone, decoder, layer_indices, family=cfg["family"],
            decoder_type=args.decoder,
        ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.decoder.load_state_dict(ckpt["decoder"])
        print(f"  Resumed decoder from {args.resume}")

    # ── Data ─────────────────────────────────────────────────────────────
    if args.mode == "train":
        splits = ["train", "val", "test"] if args.all_splits else ["train"]
        train_datasets = [CDDataset(args.data_root, split=s, crop_size=args.crop_size, augment=True,
                                    fda_target_dir=args.fda_target, fda_beta=args.fda_beta)
                          for s in splits]
        for extra in args.extra_data:
            train_datasets.append(CDDataset(extra, split="train", crop_size=args.crop_size, augment=True))
        train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        print(f"  Training samples: {len(train_ds)} ({' + '.join(str(len(d)) for d in train_datasets)})")
        val_root = args.extra_data[0] if args.all_splits and args.extra_data else args.data_root
        val_ds = CDDataset(val_root, split="val", crop_size=args.crop_size, augment=False)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        # ── Optimizer & scheduler ────────────────────────────────────────
        optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda")

        # ── Training loop ────────────────────────────────────────────────
        best_miou = 0.0
        for epoch in range(1, args.epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*60}")

            train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
            scheduler.step()

            if epoch % args.eval_every == 0 or epoch == args.epochs:
                miou, iou = evaluate(model, val_loader, device, split="val")

                ckpt_path = os.path.join(args.output_dir, f"decoder_epoch{epoch}.pth")
                torch.save({
                    "decoder": model.decoder.state_dict(),
                    "epoch": epoch,
                    "miou": miou,
                    "iou_unchanged": iou[0],
                    "iou_changed": iou[1],
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

                if miou > best_miou:
                    best_miou = miou
                    best_path = os.path.join(args.output_dir, "decoder_best.pth")
                    torch.save({
                        "decoder": model.decoder.state_dict(),
                        "epoch": epoch,
                        "miou": miou,
                        "iou_unchanged": iou[0],
                        "iou_changed": iou[1],
                    }, best_path)
                    print(f"  *** New best mIoU: {best_miou:.4f} ***")

        print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")

    # ── Test mode ────────────────────────────────────────────────────────
    if args.mode == "test" or args.mode == "train":
        if args.mode == "train":
            best_path = os.path.join(args.output_dir, "decoder_best.pth")
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                model.decoder.load_state_dict(ckpt["decoder"])
                print(f"\nLoaded best decoder (epoch {ckpt['epoch']}, mIoU={ckpt['miou']:.4f})")

        # Test on LEVIR-CD+ val
        print("\n--- Testing on LEVIR-CD+ val ---")
        val_ds = CDDataset(args.data_root, split="val", crop_size=args.crop_size, augment=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        evaluate(model, val_loader, device, split="LEVIR-CD+ val")

        # Test on S2Looking
        print("\n--- Testing on S2Looking test ---")
        s2_ds = CDDataset(
            args.test_data_root, split="test", crop_size=args.crop_size, augment=False,
            dir_a=args.test_dir_a, dir_b=args.test_dir_b,
        )
        s2_loader = DataLoader(s2_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
        evaluate(model, s2_loader, device, split="S2Looking test")


if __name__ == "__main__":
    main()
