"""Optional multi-scale spatial refinement for RPGV v5 decoder features."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .rpgv_v5_modules import ContextTopDownDecoder


class MultiScaleSpatialResidual(nn.Module):
    """Add a bounded spatial residual with an exact zero-initialized output.

    The last projection is zero-initialized and has no normalization after it.
    A v5 checkpoint therefore gives identical predictions before the first
    v5.1 optimizer update. Its first update reaches the output projection;
    spatial kernels begin receiving gradients after that projection moves.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError('channels must be at least two')
        self.pre_norm = nn.GroupNorm(math.gcd(8, channels), channels)
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.strip7 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels),
            nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels),
        )
        self.strip11 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 11), padding=(0, 5), groups=channels),
            nn.Conv2d(channels, channels, (11, 1), padding=(5, 0), groups=channels),
        )
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.pre_norm(x)
        mixed = F.gelu(self.local(normalized)
                       + self.strip7(normalized)
                       + self.strip11(normalized))
        return x + 0.5 * torch.tanh(self.output(mixed))


class SpatialResidualDecoder(ContextTopDownDecoder):
    """Preserve all v5 decoder keys and optionally refine 1/8 and 1/4."""

    def __init__(self, in_channels: Sequence[int], channels: int = 64,
                 geometry_channels: Sequence[int] = (16, 32),
                 use_context: bool = True, use_geometry: bool = True,
                 spatial_refinement: bool = True) -> None:
        super().__init__(in_channels, channels, geometry_channels,
                         use_context=use_context, use_geometry=use_geometry)
        self.spatial_refinement = bool(spatial_refinement)
        if self.spatial_refinement:
            self.spatial8 = MultiScaleSpatialResidual(channels)
            self.spatial4 = MultiScaleSpatialResidual(channels)

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
        if self.spatial_refinement:
            f8 = self.spatial8(f8)
        f4 = p4 + self._up(f8, p4.shape[-2:])
        if geometry is not None:
            f4 = self.fuse4(f4, geometry[0], reliability)
        if self.spatial_refinement:
            f4 = self.spatial4(f4)
        decoded = self.refine(f4)
        if return_pyramid:
            return decoded, (f4, f8, f16, p32)
        return decoded
