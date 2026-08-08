import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet18DeepStem, ConvModule


class UPerHead(nn.Module):
    def __init__(self, in_channels=(64, 128, 256, 512), channels=64,
                 num_classes=2, pool_scales=(1, 2, 3, 6), dropout_ratio=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        last_ch = in_channels[-1]

        self.psp_modules = nn.ModuleList()
        for scale in pool_scales:
            self.psp_modules.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(last_ch, channels, 1),
            ))
        psp_cat_ch = last_ch + channels * len(pool_scales)
        self.bottleneck = ConvModule(psp_cat_ch, channels, 3, padding=1)

        num_laterals = len(in_channels) - 1
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(num_laterals):
            self.lateral_convs.append(
                ConvModule(in_channels[i], channels, 1))
            self.fpn_convs.append(
                ConvModule(channels, channels, 3, padding=1))
        self.fpn_bottleneck = ConvModule(
            channels * len(in_channels), channels, 3, padding=1)

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def _psp_forward(self, x):
        psp_outs = [x]
        for psp in self.psp_modules:
            pooled = psp(x)
            pooled = F.interpolate(pooled, size=x.shape[2:],
                                   mode='bilinear', align_corners=False)
            psp_outs.append(pooled)
        return self.bottleneck(torch.cat(psp_outs, dim=1))

    def forward(self, inputs):
        laterals = [self.lateral_convs[i](inputs[i])
                     for i in range(len(self.lateral_convs))]
        laterals.append(self._psp_forward(inputs[-1]))

        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:],
                               mode='bilinear', align_corners=False)
            laterals[i] = laterals[i] + up

        fpn_outs = []
        for i in range(len(self.fpn_convs)):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))
        fpn_outs.append(laterals[-1])

        target_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            if fpn_outs[i].shape[2:] != target_size:
                fpn_outs[i] = F.interpolate(fpn_outs[i], size=target_size,
                                            mode='bilinear', align_corners=False)

        out = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        if self.dropout is not None:
            out = self.dropout(out)
        return self.conv_seg(out)


class SCDDecodeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.binary_cd_head = UPerHead(num_classes=2)
        self.semantic_cd_head = UPerHead(num_classes=6)
        self.semantic_cd_head_aux = UPerHead(num_classes=6)

    def forward(self, feat1, feat2):
        diff_feats = [torch.abs(f1 - f2) for f1, f2 in zip(feat1, feat2)]
        seg_logits = self.binary_cd_head(diff_feats)
        seg_logits_from = self.semantic_cd_head(feat1)
        seg_logits_to = self.semantic_cd_head_aux(feat2)
        return {
            'seg_logits': seg_logits,
            'seg_logits_from': seg_logits_from,
            'seg_logits_to': seg_logits_to,
        }


class SCDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18DeepStem()
        self.decode_head = SCDDecodeHead()
        self.is_scd = True

    def forward(self, pre, post):
        input_size = pre.shape[2:]
        feat1 = self.backbone(pre)
        feat2 = self.backbone(post)
        out = self.decode_head(feat1, feat2)
        for key in out:
            if out[key].shape[2:] != input_size:
                out[key] = F.interpolate(out[key], size=input_size,
                                         mode='bilinear', align_corners=False)
        return out
