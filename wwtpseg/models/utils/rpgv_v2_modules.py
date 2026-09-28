"""Compact decoding and a single, bounded contour correction for RPGV v2."""
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .rpgv_modules import ConvNormAct, DepthwiseSeparableBlock, binary_entropy_from_logits


class AdditiveMultiScaleDecoder(nn.Module):
    """Fuse projected scales without a 4C-wide stride-four concatenation."""

    def __init__(self, in_channels: Sequence[int], channels: int = 64):
        super().__init__()
        self.projections = nn.ModuleList([
            ConvNormAct(c, channels, kernel_size=1) for c in in_channels])
        self.refine = nn.Sequential(
            DepthwiseSeparableBlock(channels, channels),
            DepthwiseSeparableBlock(channels, channels))

    def forward(self, features):
        size = features[0].shape[-2:]
        fused = None
        for projection, feature in zip(self.projections, features):
            value = F.interpolate(projection(feature), size=size,
                                  mode='bilinear', align_corners=False)
            fused = value if fused is None else fused + value
        return self.refine(fused / len(features))


class UnifiedContourHead(nn.Module):
    """Use one signed-distance field to correct an explicitly supervised base.

    Positive SDF means foreground. The correction is bounded in logit space;
    a detached analytic uncertainty band replaces two learned spatial gates.
    Reliability controls geometry upstream, not the RGB contour head.
    """

    def __init__(self, decoder_channels=64, channels=24,
                 max_logit_correction=2.0, use_contour=True,
                 use_uncertainty_gate=True):
        super().__init__()
        if max_logit_correction <= 0:
            raise ValueError('max_logit_correction must be positive')
        self.max_logit_correction = max_logit_correction
        self.use_contour = use_contour
        self.use_uncertainty_gate = use_uncertainty_gate
        self.coarse_head = nn.Conv2d(decoder_channels, 1, 1)
        self.rgb_stem = nn.Sequential(
            ConvNormAct(3, channels), DepthwiseSeparableBlock(channels, channels))
        self.semantic_projection = ConvNormAct(decoder_channels, channels, kernel_size=1)
        self.contour = nn.Sequential(
            ConvNormAct(2 * channels + 1, channels, kernel_size=1),
            DepthwiseSeparableBlock(channels, channels),
            nn.Conv2d(channels, 1, 1))
        nn.init.zeros_(self.contour[-1].weight)
        nn.init.zeros_(self.contour[-1].bias)

    def forward(self, rgb, feature):
        coarse = self.coarse_head(feature)
        # Average before decimation reduces sensitivity to one-pixel RGB texture.
        detail = self.rgb_stem(F.avg_pool2d(rgb, 2, stride=2, ceil_mode=True))
        size = detail.shape[-2:]
        base = F.interpolate(coarse, size=size, mode='bilinear', align_corners=False)
        semantic = F.interpolate(self.semantic_projection(feature), size=size,
                                 mode='bilinear', align_corners=False)
        sdf = self.contour(torch.cat([detail, semantic, base.sigmoid()], dim=1))
        band = (binary_entropy_from_logits(base).detach()
                if self.use_uncertainty_gate else torch.ones_like(base, dtype=torch.float32))
        correction = self.max_logit_correction * band * sdf.float().tanh()
        final = base.float() + correction if self.use_contour else base.float()
        return dict(final_logits=final, coarse_logits=coarse, sdf=sdf,
                    contour_band=band, contour_correction=correction)
