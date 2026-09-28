"""Top-down RGB decoding with late, confidence-gated geometry correction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .rpgv_modules import ConvNormAct, DepthwiseSeparableBlock


class CompactGeometryEncoder(nn.Module):
    """Encode depth only after invalid pixels have been masked in image space."""

    def __init__(self, channels: Sequence[int] = (16, 32)) -> None:
        super().__init__()
        if len(channels) != 2 or min(channels) < 2:
            raise ValueError('geometry channels must contain two values >= 2')
        self.stem = nn.Sequential(
            ConvNormAct(2, channels[0], stride=2),
            DepthwiseSeparableBlock(channels[0], channels[0], stride=2),
        )
        self.down = DepthwiseSeparableBlock(channels[0], channels[1], stride=2)

    def forward(self, depth: torch.Tensor, reliability: torch.Tensor):
        if depth.shape != reliability.shape:
            raise ValueError('depth and reliability must have matching shapes')
        # Mask before any spatial convolution. Values at Q=0 cannot influence
        # a neighbouring valid feature, regardless of their original depth.
        masked = depth * reliability
        quarter = self.stem(torch.cat([masked, reliability], dim=1))
        eighth = self.down(quarter)
        return quarter, eighth


class BoundedGeometryFusion(nn.Module):
    """Bounded decoder residual with a nonzero first-step geometry gradient."""

    def __init__(self, rgb_channels: int, geometry_channels: int,
                 initial_scale: float = 0.1, max_scale: float = 0.5) -> None:
        super().__init__()
        if not 0 < initial_scale < max_scale:
            raise ValueError('initial_scale must lie strictly between 0 and max_scale')
        self.max_scale = float(max_scale)
        self.rgb_projection = ConvNormAct(rgb_channels, geometry_channels, kernel_size=1)
        # The last projection has no normalization or zero initialization.
        self.output = nn.Conv2d(3 * geometry_channels, rgb_channels, 1)
        fraction = initial_scale / max_scale
        self.scale_logit = nn.Parameter(torch.tensor(math.log(fraction / (1 - fraction))))

    def forward(self, rgb: torch.Tensor, geometry: torch.Tensor,
                reliability: torch.Tensor) -> torch.Tensor:
        size = rgb.shape[-2:]
        if geometry.shape[-2:] != size:
            geometry = F.interpolate(geometry, size=size, mode='bilinear', align_corners=False)
        q = F.adaptive_avg_pool2d(reliability.float(), size).to(rgb.dtype)
        projected = self.rgb_projection(rgb)
        residual = torch.tanh(self.output(torch.cat(
            [projected, geometry, projected * geometry], dim=1)))
        scale = self.max_scale * torch.sigmoid(self.scale_logit)
        return rgb + scale * q * residual


class ContextTopDownDecoder(nn.Module):
    """Pool C4 context once, then fuse C4→C1 with optional late geometry."""

    def __init__(self, in_channels: Sequence[int], channels: int = 64,
                 geometry_channels: Sequence[int] = (16, 32),
                 use_context: bool = True, use_geometry: bool = True) -> None:
        super().__init__()
        if len(in_channels) != 4 or channels < 2:
            raise ValueError('decoder requires four RGB stages and channels >= 2')
        self.projections = nn.ModuleList([
            ConvNormAct(in_channel, channels, kernel_size=1)
            for in_channel in in_channels
        ])
        self.use_context = bool(use_context)
        self.use_geometry = bool(use_geometry)
        if self.use_context:
            self.context_projections = nn.ModuleList([
                ConvNormAct(in_channels[-1], channels, kernel_size=1)
                for _ in (1, 2, 4)
            ])
        if self.use_geometry:
            if len(geometry_channels) != 2:
                raise ValueError('decoder requires geometry channels for 1/4 and 1/8')
            self.fuse8 = BoundedGeometryFusion(channels, geometry_channels[1])
            self.fuse4 = BoundedGeometryFusion(channels, geometry_channels[0])
        self.refine = nn.Sequential(
            DepthwiseSeparableBlock(channels, channels),
            DepthwiseSeparableBlock(channels, channels),
        )

    @staticmethod
    def _up(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(value, size=size, mode='bilinear', align_corners=False)

    def forward(self, features: Sequence[torch.Tensor],
                geometry: tuple[torch.Tensor, torch.Tensor] | None = None,
                reliability: torch.Tensor | None = None,
                return_pyramid: bool = False):
        if len(features) != 4:
            raise ValueError('decoder expects four RGB feature maps')
        p4, p8, p16, p32 = [
            projection(feature)
            for projection, feature in zip(self.projections, features)
        ]
        if self.use_context:
            context = []
            for grid, projection in zip((1, 2, 4), self.context_projections):
                pooled = F.adaptive_avg_pool2d(features[-1], (
                    min(grid, features[-1].shape[-2]),
                    min(grid, features[-1].shape[-1])))
                context.append(self._up(projection(pooled), p32.shape[-2:]))
            p32 = (p32 + sum(context)) / (1 + len(context))
        f16 = p16 + self._up(p32, p16.shape[-2:])
        f8 = p8 + self._up(f16, p8.shape[-2:])
        if geometry is not None:
            if not self.use_geometry or reliability is None:
                raise ValueError('geometry requires an enabled fusion and reliability')
            f8 = self.fuse8(f8, geometry[1], reliability)
        f4 = p4 + self._up(f8, p4.shape[-2:])
        if geometry is not None:
            f4 = self.fuse4(f4, geometry[0], reliability)
        decoded = self.refine(f4)
        if return_pyramid:
            return decoded, (f4, f8, f16, p32)
        return decoded
