"""DFormer-inspired compact RGB/geometry interaction (not a DFormer port)."""
import torch
from torch import nn
from torch.nn import functional as F

from .rpgv_modules import ConvNormAct, DepthwiseSeparableBlock, HaarWavelet2D


class GeometryFusionBlock(nn.Module):
    """Local multiplicative interaction, Haar bands and pooled joint queries.

    The segmentation stream and the next geometry stage share joint features.
    Offline confidence gates both streams; there is no corrected depth target,
    independent geometry segmentation head, or zero-initialized fusion barrier.
    """

    def __init__(self, rgb_channels, geometry_channels, use_frequency=True,
                 context_grid=0, update_geometry=True):
        super().__init__()
        c = geometry_channels
        self.rgb_projection = ConvNormAct(rgb_channels, c, kernel_size=1)
        self.geometry_local = DepthwiseSeparableBlock(c, c)
        self.use_frequency = use_frequency
        self.context_grid = context_grid
        self.update_geometry = update_geometry
        joint_channels = 3 * c
        if use_frequency:
            self.haar = HaarWavelet2D()
            self.frequency_projection = ConvNormAct(8 * c, c, kernel_size=1)
            joint_channels += c
        if context_grid:
            self.query = nn.Conv2d(2 * c, c, 1)
            self.key_value = nn.Conv2d(c, 2 * c, 1)
            joint_channels += c
        self.mix = nn.Sequential(
            ConvNormAct(joint_channels, c, kernel_size=1),
            DepthwiseSeparableBlock(c, c))
        self.rgb_output = nn.Conv2d(c, rgb_channels, 1)
        self.rgb_scale = nn.Parameter(torch.full((1, rgb_channels, 1, 1), 0.1))
        if update_geometry:
            self.geometry_output = nn.Conv2d(c, c, 1)

    def forward(self, rgb, geometry, reliability):
        size = rgb.shape[-2:]
        q = F.adaptive_avg_pool2d(reliability.float(), size).to(rgb.dtype)
        r = self.rgb_projection(rgb)
        g = self.geometry_local(geometry) * q
        parts = [r, g, r * g]
        if self.use_frequency:
            r_low, r_high = self.haar(r)
            g_low, g_high = self.haar(g)
            bands = self.frequency_projection(torch.cat(
                [r_low, g_low, r_high, g_high], dim=1))
            parts.append(F.interpolate(bands, size=size, mode='bilinear',
                                       align_corners=False))
        if self.context_grid:
            grid = (min(self.context_grid, size[0]), min(self.context_grid, size[1]))
            query = self.query(F.adaptive_avg_pool2d(torch.cat([r, g], 1), grid))
            key, value = self.key_value(r).chunk(2, dim=1)
            # At most grid^2 x HW scores; FP32 softmax/matmul under AMP.
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                attention = (query.float().flatten(2).transpose(1, 2)
                             @ key.float().flatten(2)) * r.shape[1] ** -0.5
                context = attention.softmax(-1) @ value.float().flatten(2).transpose(1, 2)
            context = context.transpose(1, 2).reshape(rgb.shape[0], -1, *grid)
            parts.append(F.interpolate(context.to(r.dtype), size=size,
                                       mode='bilinear', align_corners=False))
        joint = self.mix(torch.cat(parts, dim=1))
        fused_rgb = rgb + self.rgb_scale * q * self.rgb_output(joint)
        fused_geometry = ((geometry + self.geometry_output(joint)) * q
                          if self.update_geometry else geometry)
        return fused_rgb, fused_geometry
