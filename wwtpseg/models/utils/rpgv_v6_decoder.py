"""V6-only decoder fusion and independently supervised contour refinement."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .rpgv_v51_modules import SpatialResidualDecoder
from .rpgv_v6_modules import _conv_block


class SpatialGeometryDecoderV6(SpatialResidualDecoder):
    """Reuse the spatial decoder's parameters, injecting geometry before F8/F4.

    The callback receives the current feature and a geometry-free lateral RGB
    reference. In particular, P4 must not be replaced with F4 for InfoNCE: F4
    already contains the geometry propagated through F8.
    """

    def __init__(self, in_channels, channels=64, geometry_channels=(32, 64),
                 use_context=True, spatial_refinement=True):
        super().__init__(in_channels, channels, geometry_channels,
                         use_context=use_context, use_geometry=False,
                         spatial_refinement=spatial_refinement)
        for module in self.modules():
            if isinstance(module, nn.GELU):
                module.approximate = 'tanh'

    @staticmethod
    def _spatial(block, value):
        normalized = block.pre_norm(value)
        mixed = F.gelu(block.local(normalized) + block.strip7(normalized)
                       + block.strip11(normalized), approximate='tanh')
        return value + 0.5 * torch.tanh(block.output(mixed))

    def forward(self, features, fusion_callback=None, return_pyramid=False):
        if len(features) != 4:
            raise ValueError('decoder expects four RGB feature maps')
        p4, p8, p16, p32 = [projection(feature)
                            for projection, feature in zip(self.projections, features)]
        if self.use_context:
            context = []
            for grid, projection in zip((1, 2, 4), self.context_projections):
                pooled = F.adaptive_avg_pool2d(features[-1], (
                    min(grid, features[-1].shape[-2]), min(grid, features[-1].shape[-1])))
                context.append(self._up(projection(pooled), p32.shape[-2:]))
            p32 = (p32 + sum(context)) / (1 + len(context))
        f16 = p16 + self._up(p32, p16.shape[-2:])
        f8 = p8 + self._up(f16, p8.shape[-2:])
        if fusion_callback is not None:
            f8 = fusion_callback(1, f8, p8)
        if self.spatial_refinement:
            f8 = self._spatial(self.spatial8, f8)
        f4 = p4 + self._up(f8, p4.shape[-2:])
        if fusion_callback is not None:
            f4 = fusion_callback(0, f4, p4)
        if self.spatial_refinement:
            f4 = self._spatial(self.spatial4, f4)
        decoded = f4
        for block in self.refine:
            decoded = F.gelu(block.block(decoded) + block.shortcut(decoded), approximate='tanh')
        if return_pyramid:
            return decoded, (f4, f8, f16, p32)
        return decoded


class _DepthwiseResidualV6(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(math.gcd(8, channels), channels), nn.GELU(approximate='tanh'))
        self.projection = _conv_block(channels, channels, kernel_size=1)

    def forward(self, value):
        return F.gelu(value + self.projection(self.spatial(value)), approximate='tanh')


class ResidualContourHeadV6(nn.Module):
    """SDF is auxiliary; an unrestricted learned logit residual refines the mask."""

    def __init__(self, decoder_channels=64, channels=24, use_contour=True, use_sdf=True):
        super().__init__()
        self.use_contour = bool(use_contour)
        self.use_sdf = bool(use_sdf)
        self.coarse_head = nn.Conv2d(decoder_channels, 1, 1)
        if self.use_contour or self.use_sdf:
            self.rgb_stem = nn.Sequential(_conv_block(3, channels), _DepthwiseResidualV6(channels))
            self.semantic_projection = _conv_block(decoder_channels, channels, kernel_size=1)
            self.contour = nn.Sequential(
                _conv_block(2 * channels + 1, channels, kernel_size=1), _DepthwiseResidualV6(channels))
        if self.use_sdf:
            self.sdf_head = nn.Conv2d(channels, 1, 1)
            nn.init.normal_(self.sdf_head.weight, std=1e-3)
            nn.init.zeros_(self.sdf_head.bias)
        if self.use_contour:
            self.correction_head = nn.Conv2d(channels, 1, 1)
            nn.init.normal_(self.correction_head.weight, std=1e-3)
            nn.init.zeros_(self.correction_head.bias)

    def forward(self, rgb, feature):
        coarse = self.coarse_head(feature)
        size = ((rgb.shape[-2] + 1) // 2, (rgb.shape[-1] + 1) // 2)
        base = F.interpolate(coarse, size=size, mode='bilinear', align_corners=False)
        if self.use_contour or self.use_sdf:
            detail = self.rgb_stem(F.avg_pool2d(rgb, 2, stride=2, ceil_mode=True))
            semantic = F.interpolate(self.semantic_projection(feature), size=size,
                                     mode='bilinear', align_corners=False)
            hidden = self.contour(torch.cat((detail, semantic, base.sigmoid()), dim=1).contiguous())
        sdf = (self.sdf_head(hidden) if self.use_sdf else torch.zeros_like(base))
        correction = (self.correction_head(hidden).float() if self.use_contour
                      else torch.zeros_like(base, dtype=torch.float32))
        return dict(final_logits=base.float() + correction, coarse_logits=coarse, sdf=sdf,
                    contour_band=torch.ones_like(base, dtype=torch.float32),
                    contour_correction=correction)
