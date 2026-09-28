"""Small geometry-conditioned error corrector for a frozen RGB predictor."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .rpgv_modules import ConvNormAct, DepthwiseSeparableBlock, spatial_derivatives


class GeometryErrorAdapter(nn.Module):
    """Read geometry at three scales and predict a bounded logit correction.

    The output projection has no normalization after it. Zero initialization
    therefore provides both exact RGB initialization and a small first update.
    """

    def __init__(self, semantic_channels=64, channels=24, max_correction=3.0):
        super().__init__()
        if channels < 2 or not math.isfinite(max_correction) or max_correction <= 0:
            raise ValueError('channels >=2 and finite max_correction >0 are required')
        self.max_correction = max_correction
        self.geometry_stem = nn.Sequential(
            ConvNormAct(8, channels), DepthwiseSeparableBlock(channels, channels))
        self.down = nn.ModuleList([
            DepthwiseSeparableBlock(channels, channels, stride=2) for _ in range(2)])
        self.semantic = ConvNormAct(semantic_channels, channels, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvNormAct(2 * channels + 2, channels, kernel_size=1),
            DepthwiseSeparableBlock(channels, channels))
        self.output = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, depth, quality, semantic, base_probability, uncertainty):
        size = semantic.shape[-2:]
        depth = F.interpolate(depth, size=size, mode='bilinear', align_corners=False)
        quality = F.interpolate(quality, size=size, mode='bilinear', align_corners=False)
        dx, dy, magnitude, curvature = spatial_derivatives(depth)
        # Replication avoids manufacturing depth discontinuities at crop edges.
        relief = [depth - F.avg_pool2d(F.pad(depth, (k//2,) * 4, mode='replicate'), k, 1)
                  for k in (3, 7)]
        descriptor = torch.cat([depth, dx, dy, magnitude, curvature, *relief, quality], 1)
        feature = self.geometry_stem(descriptor)
        context = feature
        for downsample in self.down:
            context = downsample(context)
            feature = feature + F.interpolate(context, size=size, mode='bilinear', align_corners=False)
        probability = F.interpolate(base_probability, size=size, mode='bilinear', align_corners=False)
        uncertainty = F.interpolate(uncertainty, size=size, mode='bilinear', align_corners=False)
        feature = self.fuse(torch.cat([feature / 3, self.semantic(semantic), probability, uncertainty], 1))
        return self.max_correction * self.output(feature).float().tanh()
