import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet18DeepStem, ConvModule


class TwoIdentity(nn.Module):
    def forward(self, x1, x2):
        return x1, x2


class SpatialExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w, device=x1.device) % self.p == 0
        out_x1 = x1.clone()
        out_x2 = x2.clone()
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]
        return out_x1, out_x2


class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_map = torch.arange(c, device=x1.device) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand(N, -1)
        out_x1 = torch.zeros_like(x1)
        out_x2 = torch.zeros_like(x2)
        out_x1[~exchange_mask] = x1[~exchange_mask]
        out_x2[~exchange_mask] = x2[~exchange_mask]
        out_x1[exchange_mask] = x2[exchange_mask]
        out_x2[exchange_mask] = x1[exchange_mask]
        return out_x1, out_x2


class IAResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResNet18DeepStem._make_layer(64, 64, 2, stride=1)
        self.layer2 = ResNet18DeepStem._make_layer(64, 128, 2, stride=2)
        self.layer3 = ResNet18DeepStem._make_layer(128, 256, 2, stride=2)
        self.layer4 = ResNet18DeepStem._make_layer(256, 512, 2, stride=2)
        self.ccs = nn.ModuleList([
            TwoIdentity(),
            SpatialExchange(p=2),
            ChannelExchange(p=2),
            ChannelExchange(p=2),
        ])

    def _stem_forward(self, x):
        x = self.relu(self.stem(x))
        x = self.maxpool(x)
        return x

    def forward(self, x1, x2):
        x1 = self._stem_forward(x1)
        x2 = self._stem_forward(x2)
        outs = []
        for i, layer in enumerate([self.layer1, self.layer2,
                                    self.layer3, self.layer4]):
            x1 = layer(x1)
            x2 = layer(x2)
            x1, x2 = self.ccs[i](x1, x2)
            outs.append(torch.cat([x1, x2], dim=1))
        return outs


class FDAF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        c2 = in_channels * 2
        self.flow_make = nn.Sequential(
            nn.Conv2d(c2, c2, 5, padding=2, groups=c2, bias=True),
            nn.InstanceNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, 4, 1, bias=False),
        )

    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()
        norm = torch.tensor([[[[w, h]]]], dtype=x.dtype, device=x.device)
        col = torch.linspace(-1.0, 1.0, h, device=x.device).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w, device=x.device).repeat(h, 1)
        grid = torch.stack([row, col], dim=2)
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        return F.grid_sample(x, grid, align_corners=True)

    def forward(self, x1, x2):
        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        f1, f2 = torch.chunk(flow, 2, dim=1)
        x1_feat = self.warp(x1, f1) - x2
        x2_feat = self.warp(x2, f2) - x1
        return torch.cat([x1_feat, x2_feat], dim=1)


class MixFFN(nn.Module):
    def __init__(self, embed_dims, feedforward_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(embed_dims, feedforward_channels, 1, bias=True),
            nn.Conv2d(feedforward_channels, feedforward_channels, 3,
                      padding=1, groups=feedforward_channels, bias=True),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Conv2d(feedforward_channels, embed_dims, 1, bias=True),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        return x + self.layers(x)


class ChangerHead(nn.Module):
    def __init__(self, in_channels=(64, 128, 256, 512), channels=128):
        super().__init__()
        num_inputs = len(in_channels)
        self.convs = nn.ModuleList()
        for in_ch in in_channels:
            self.convs.append(ConvModule(in_ch, channels, 1))
        self.fusion_conv = ConvModule(channels * num_inputs,
                                      channels // 2, 1)
        self.neck_layer = FDAF(in_channels=channels // 2)
        self.discriminator = MixFFN(channels, channels)
        self.conv_seg = nn.Conv2d(channels, 2, 1)

    def _base_forward(self, inputs):
        target_size = inputs[0].shape[2:]
        outs = []
        for idx, x in enumerate(inputs):
            out = self.convs[idx](x)
            if out.shape[2:] != target_size:
                out = F.interpolate(out, size=target_size,
                                    mode='bilinear', align_corners=False)
            outs.append(out)
        return self.fusion_conv(torch.cat(outs, dim=1))

    def forward(self, features):
        inputs1, inputs2 = [], []
        for feat in features:
            f1, f2 = torch.chunk(feat, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)
        out1 = self._base_forward(inputs1)
        out2 = self._base_forward(inputs2)
        out = self.neck_layer(out1, out2)
        out = self.discriminator(out)
        return self.conv_seg(out)


class ChangerExModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = IAResNet18()
        self.decode_head = ChangerHead()

    def forward(self, pre, post):
        input_size = pre.shape[2:]
        features = self.backbone(pre, post)
        logits = self.decode_head(features)
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size,
                                   mode='bilinear', align_corners=False)
        return logits
