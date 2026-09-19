"""Compact MMSegmentation integration of the RS-Mamba segmentation encoder.

The implementation keeps RS-Mamba's eight-direction recurrent state-space
scan and U-shaped decoder while exposing the decoded stride-4 feature to an
ordinary MMSeg decode head. CUDA training uses mamba-ssm's fused selective
scan. A differentiable PyTorch reference scan is retained for tiny CPU tests;
it is intentionally not suitable for full-resolution training.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import torch
import torch.utils.checkpoint
from einops import repeat
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from timm.layers import DropPath, trunc_normal_
from torch import nn
from torch.nn import functional as F


def _diagonal_gather(x: torch.Tensor, anti: bool = False) -> torch.Tensor:
    batch, channels, height, width = x.shape
    rows = torch.arange(height, device=x.device).unsqueeze(1)
    columns = torch.arange(width, device=x.device)
    index = (columns - rows) % width if anti else (columns + rows) % width
    index = index[None, None].expand(batch, channels, -1, -1)
    return x.gather(3, index).transpose(-1, -2).reshape(batch, channels, -1)


def _diagonal_scatter(
    sequence: torch.Tensor,
    height: int,
    width: int,
    anti: bool = False,
) -> torch.Tensor:
    batch, channels, _ = sequence.shape
    rows = torch.arange(height, device=sequence.device).unsqueeze(1)
    columns = torch.arange(width, device=sequence.device)
    index = (columns - rows) % width if anti else (columns + rows) % width
    index = index[None, None].expand(batch, channels, -1, -1)
    values = sequence.reshape(batch, channels, width, height).transpose(-1, -2)
    return torch.zeros(
        batch, channels, height, width,
        device=sequence.device, dtype=sequence.dtype).scatter(3, index, values)


def _make_directions(x: torch.Tensor) -> torch.Tensor:
    row = x.flatten(2)
    column = x.transpose(2, 3).contiguous().flatten(2)
    diagonal = _diagonal_gather(x)
    anti_diagonal = _diagonal_gather(x, anti=True)
    forward = torch.stack([row, column, diagonal, anti_diagonal], dim=1)
    return torch.cat([forward, forward.flip(-1)], dim=1)


def _merge_directions(
    sequences: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Undo each directional traversal and sum its response."""
    row = sequences[:, 0] + sequences[:, 4].flip(-1)
    column = sequences[:, 1] + sequences[:, 5].flip(-1)
    column = column.view(
        column.shape[0], column.shape[1], width, height)
    column = column.transpose(2, 3).contiguous().flatten(2)
    diagonal = sequences[:, 2] + sequences[:, 6].flip(-1)
    diagonal = _diagonal_scatter(diagonal, height, width).flatten(2)
    anti = sequences[:, 3] + sequences[:, 7].flip(-1)
    anti = _diagonal_scatter(anti, height, width, anti=True).flatten(2)
    return row + column + diagonal + anti


@lru_cache(maxsize=1)
def _fused_selective_scan():
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        return selective_scan_fn
    except ImportError:
        return None


def _reference_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    delta_bias: torch.Tensor,
) -> torch.Tensor:
    """Small-input PyTorch reference for environments without CUDA kernels."""
    batch, kd, length = u.shape
    groups, state_size = b.shape[1:3]
    channels = kd // groups
    u = u.view(batch, groups, channels, length)
    delta = F.softplus(
        delta.view(batch, groups, channels, length)
        + delta_bias.view(1, groups, channels, 1))
    a = a.view(groups, channels, state_size)
    d = d.view(groups, channels)
    state = u.new_zeros(batch, groups, channels, state_size)
    outputs = []
    for index in range(length):
        dt = delta[..., index]
        decay = torch.exp(dt[..., None] * a[None])
        drive = (
            dt[..., None] * u[..., index, None]
            * b[:, :, None, :, index])
        state = decay * state + drive
        output = (state * c[:, :, None, :, index]).sum(-1)
        outputs.append(output + d[None] * u[..., index])
    return torch.stack(outputs, dim=-1).reshape(batch, kd, length)


class OmniSelectiveScan2D(nn.Module):
    """Eight-direction selective state-space scan from RS-Mamba."""

    def __init__(
        self,
        dim: int,
        state_size: int = 16,
        expansion: float = 2.0,
        dt_rank: int | str = 'auto',
        conv_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.inner_dim = int(dim * expansion)
        self.state_size = state_size
        self.dt_rank = math.ceil(dim / 16) if dt_rank == 'auto' else int(dt_rank)
        self.num_directions = 8

        self.in_proj = nn.Linear(dim, 2 * self.inner_dim, bias=False)
        self.depthwise = nn.Conv2d(
            self.inner_dim, self.inner_dim, conv_size,
            padding=conv_size // 2, groups=self.inner_dim, bias=True)
        projection_size = self.dt_rank + 2 * state_size
        projections = [
            nn.Linear(self.inner_dim, projection_size, bias=False)
            for _ in range(self.num_directions)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([layer.weight for layer in projections]))
        dt_projections = [self._make_dt_projection() for _ in range(8)]
        self.dt_proj_weight = nn.Parameter(
            torch.stack([layer.weight for layer in dt_projections]))
        self.dt_proj_bias = nn.Parameter(
            torch.stack([layer.bias for layer in dt_projections]))
        self.a_logs = self._init_a_logs()
        self.ds = nn.Parameter(torch.ones(8 * self.inner_dim))
        self.output_norm = nn.LayerNorm(self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def _make_dt_projection(self) -> nn.Linear:
        layer = nn.Linear(self.dt_rank, self.inner_dim, bias=True)
        bound = self.dt_rank**-0.5
        nn.init.uniform_(layer.weight, -bound, bound)
        dt = torch.exp(
            torch.rand(self.inner_dim)
            * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            layer.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        return layer

    def _init_a_logs(self) -> nn.Parameter:
        values = repeat(
            torch.arange(1, self.state_size + 1, dtype=torch.float32),
            'n -> d n', d=self.inner_dim)
        return nn.Parameter(
            repeat(values.log(), 'd n -> k d n', k=8).flatten(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, _ = x.shape
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        gate = F.silu(gate)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.silu(self.depthwise(x))
        directions = _make_directions(x)
        projected = torch.einsum(
            'b k d l, k c d -> b k c l', directions, self.x_proj_weight)
        dt, b, c = torch.split(
            projected, [self.dt_rank, self.state_size, self.state_size], dim=2)
        dt = torch.einsum(
            'b k r l, k d r -> b k d l', dt, self.dt_proj_weight)
        length = height * width
        u = directions.reshape(batch, -1, length).contiguous()
        dt = dt.reshape(batch, -1, length).contiguous()
        a = -self.a_logs.float().exp()

        fused_scan = _fused_selective_scan() if x.is_cuda else None
        if fused_scan is not None:
            y = fused_scan(
                u.float(), dt.float(), a, b.float(), c.float(),
                self.ds.float(), z=None,
                delta_bias=self.dt_proj_bias.float().reshape(-1),
                delta_softplus=True,
                return_last_state=False,
            ).to(x.dtype)
        else:
            y = _reference_selective_scan(
                u.float(), dt.float(), a, b.float(), c.float(),
                self.ds.float(), self.dt_proj_bias.float().reshape(-1),
            ).to(x.dtype)

        y = y.view(batch, 8, self.inner_dim, height * width)
        y = _merge_directions(y, height, width).transpose(1, 2)
        y = self.output_norm(y).view(batch, height, width, self.inner_dim)
        return self.dropout(self.out_proj(y * gate))


class ChannelMlp(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RSMBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        state_size: int,
        ssm_ratio: float,
        mlp_ratio: float,
        drop_path: float,
        use_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.scan = OmniSelectiveScan2D(dim, state_size, ssm_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ChannelMlp(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.use_checkpoint = use_checkpoint

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.scan(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False)
        return self._forward(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


@MODELS.register_module()
class RSMambaBackbone(BaseModule):
    """RS-Mamba encoder-decoder returning a stride-4 decoded feature map."""

    def __init__(
        self,
        in_channels: int = 3,
        dims: int | Sequence[int] = 96,
        depths: Sequence[int] = (2, 2, 9, 2),
        state_size: int = 16,
        ssm_ratio: float = 2.0,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        use_checkpoint: bool = False,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if isinstance(dims, int):
            dims = tuple(dims * 2**index for index in range(len(depths)))
        if len(dims) != len(depths):
            raise ValueError('dims and depths must have the same length')
        self.dims = tuple(dims)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, self.dims[0] // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(self.dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(self.dims[0] // 2, self.dims[0], 3, stride=2, padding=1),
        )
        self.patch_norm = nn.LayerNorm(self.dims[0])
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        downsamples = [nn.Identity()] + [
            Downsample(self.dims[index - 1], self.dims[index])
            for index in range(1, len(self.dims))
        ]
        stages = []
        offset = 0
        for dim, depth, downsample in zip(self.dims, depths, downsamples):
            blocks = [
                RSMBlock(
                    dim, state_size, ssm_ratio, mlp_ratio,
                    rates[offset + index], use_checkpoint)
                for index in range(depth)
            ]
            stages.append(nn.Sequential(downsample, *blocks))
            offset += depth
        self.stages = nn.ModuleList(stages)
        self.decoders = nn.ModuleList([
            DecoderBlock(self.dims[index], self.dims[index - 1])
            for index in range(len(self.dims) - 1, 0, -1)
        ])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        x = self.patch_norm(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x.permute(0, 3, 1, 2).contiguous())
        decoded = features[-1]
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            decoded = decoder(decoded, skip)
        return (decoded, )
