import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Minimal ViT-B/14 (DINOv2-compatible, no timm dependency)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4, init_values=1e-5):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Identity()
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, init_values) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def _interpolate_pos_embed(self, num_patches):
        N = self.pos_embed.shape[1] - 1
        if num_patches == N:
            return self.pos_embed
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = patch_pos.shape[-1]
        orig_grid = int(N ** 0.5)
        new_grid = int(num_patches ** 0.5)
        patch_pos = patch_pos.reshape(1, orig_grid, orig_grid, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos.float(), size=(new_grid, new_grid),
            mode='bicubic', align_corners=False,
        ).to(patch_pos.dtype)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x):
        x = self.patch_embed(x)
        pos_embed = self._interpolate_pos_embed(x.shape[1])
        x = x + pos_embed[:, 1:]
        cls = (self.cls_token + pos_embed[:, :1]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 1:]


# ---------------------------------------------------------------------------
# Change detection decoder & detector
# ---------------------------------------------------------------------------

def _conv_bn_relu(in_ch, out_ch, kernel=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MultiLayerChangeDecoder(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, out_size=518,
                 patch_grid=37, num_layers=4):
        super().__init__()
        self.patch_grid = patch_grid
        self.out_size = out_size
        self.num_layers = num_layers

        self.layer_norms = nn.ModuleList([nn.LayerNorm(in_dim) for _ in range(num_layers)])
        self.layer_projs = nn.ModuleList([
            _conv_bn_relu(in_dim * 3, hidden_dim, kernel=1, padding=0)
            for _ in range(num_layers)
        ])

        self.upsample_blocks = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        for _ in range(3):
            self.upsample_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ))
            self.fuse_convs.append(nn.Sequential(
                _conv_bn_relu(hidden_dim * 2, hidden_dim),
                _conv_bn_relu(hidden_dim, hidden_dim),
            ))

        self.head = nn.Sequential(
            _conv_bn_relu(hidden_dim, hidden_dim // 2),
            _conv_bn_relu(hidden_dim // 2, hidden_dim // 4),
            nn.Conv2d(hidden_dim // 4, 2, 1),
        )

    def forward(self, feats_a, feats_b, out_size=None):
        B, N, _ = feats_a[0].shape
        H = W = int(N ** 0.5)

        projected = []
        for i in range(self.num_layers):
            fa_normed = self.layer_norms[i](feats_a[i])
            fb_normed = self.layer_norms[i](feats_b[i])
            fa = fa_normed.permute(0, 2, 1).reshape(B, -1, H, W)
            fb = fb_normed.permute(0, 2, 1).reshape(B, -1, H, W)
            diff = torch.abs(fa - fb)
            x = torch.cat([fa, fb, diff], dim=1)
            projected.append(self.layer_projs[i](x))

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


class DINOv2ChangeDetector(nn.Module):
    def __init__(self, backbone, decoder, layer_indices):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.layer_indices = layer_indices

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def extract_multilayer_features(self, x):
        x = self.backbone.patch_embed(x)
        pos_embed = self.backbone._interpolate_pos_embed(x.shape[1])
        x = x + pos_embed[:, 1:]
        cls_token = self.backbone.cls_token + pos_embed[:, :1]
        x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.backbone.pos_drop(x)

        feats = []
        for idx, block in enumerate(self.backbone.blocks):
            x = block(x)
            if idx in self.layer_indices:
                feats.append(x[:, 1:])
        return feats

    def forward(self, img_a, img_b):
        out_size = img_a.shape[2:]
        with torch.no_grad():
            feats_a = self.extract_multilayer_features(img_a)
            feats_b = self.extract_multilayer_features(img_b)
        return self.decoder(feats_a, feats_b, out_size=out_size)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
