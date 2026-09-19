"""UNetFormer backbone/decoder adapted to the MMSegmentation interface.

The global-local attention, weighted feature fusion and feature-refinement
head follow the official GeoSeg UNetFormer implementation. Classification is
left to MMSeg decode heads so that losses and ablations remain configurable.
"""

from __future__ import annotations

import torch
from einops import rearrange
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from timm.layers import DropPath, trunc_normal_
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        activation: bool = True,
    ) -> None:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class SeparableConvBN(nn.Sequential):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__(
            nn.Conv2d(
                channels, channels, kernel_size,
                padding=(kernel_size - 1) // 2,
                groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )


class GlobalLocalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 8):
        super().__init__()
        if dim % num_heads:
            raise ValueError('dim must be divisible by num_heads')
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads)**-0.5
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=False)
        self.local3 = ConvBNAct(dim, dim, activation=False)
        self.local1 = ConvBNAct(dim, dim, kernel_size=1, activation=False)
        self.proj = SeparableConvBN(dim, kernel_size=window_size)
        self.pool_x = nn.AvgPool2d(
            (window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.pool_y = nn.AvgPool2d(
            (1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        table_size = (2 * window_size - 1)**2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size),
            indexing='ij'))
        flat = coords.flatten(1)
        relative = flat[:, :, None] - flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer(
            'relative_position_index', relative.sum(-1), persistent=False)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    @staticmethod
    def _pad_to_window(x: torch.Tensor, window_size: int) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = (-height) % window_size
        pad_w = (-width) % window_size
        if pad_h or pad_w:
            mode = 'reflect' if pad_h < height and pad_w < width else 'replicate'
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        local = self.local3(x) + self.local1(x)
        padded = self._pad_to_window(x, self.window_size)
        padded_h, padded_w = padded.shape[-2:]
        q, k, v = rearrange(
            self.qkv(padded),
            'b (qkv head d) (nh wh) (nw ww) -> qkv (b nh nw) head (wh ww) d',
            qkv=3,
            head=self.num_heads,
            d=channels // self.num_heads,
            nh=padded_h // self.window_size,
            nw=padded_w // self.window_size,
            wh=self.window_size,
            ww=self.window_size,
        )
        attention = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)]
        bias = bias.view(
            self.window_size**2, self.window_size**2,
            self.num_heads).permute(2, 0, 1)
        attention = (attention + bias.unsqueeze(0)).softmax(dim=-1) @ v
        attention = rearrange(
            attention,
            '(b nh nw) head (wh ww) d -> b (head d) (nh wh) (nw ww)',
            b=batch,
            nh=padded_h // self.window_size,
            nw=padded_w // self.window_size,
            wh=self.window_size,
            ww=self.window_size,
        )[:, :, :height, :width]
        pooled = self.pool_x(F.pad(attention, (0, 0, 0, 1), mode='reflect'))
        pooled += self.pool_y(F.pad(attention, (0, 1, 0, 0), mode='reflect'))
        fused = F.pad(pooled + local, (0, 1, 0, 1), mode='reflect')
        return self.proj(fused)[:, :, :height, :width]


class SpatialMlp(nn.Module):
    def __init__(self, dim: int, expansion: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.layers = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GlobalLocalBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, window_size: int, drop_path: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attention = GlobalLocalAttention(dim, num_heads, window_size)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = SpatialMlp(dim)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class WeightedFusion(nn.Module):
    def __init__(self, skip_channels: int, decode_channels: int):
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_channels, decode_channels, 1, bias=False)
        self.weights = nn.Parameter(torch.ones(2))
        self.post = ConvBNAct(decode_channels, decode_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + 1e-8)
        return self.post(weights[0] * self.skip_proj(skip) + weights[1] * x)


class FeatureRefinementHead(WeightedFusion):
    def __init__(self, skip_channels: int, decode_channels: int):
        super().__init__(skip_channels, decode_channels)
        self.pixel_attention = nn.Sequential(
            nn.Conv2d(
                decode_channels, decode_channels, 3, padding=1,
                groups=decode_channels),
            nn.Sigmoid(),
        )
        squeeze_channels = max(1, decode_channels // 16)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(decode_channels, squeeze_channels, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(squeeze_channels, decode_channels, 1),
            nn.Sigmoid(),
        )
        self.shortcut = ConvBNAct(
            decode_channels, decode_channels, kernel_size=1, activation=False)
        self.proj = SeparableConvBN(decode_channels)
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = super().forward(x, skip)
        attended = self.pixel_attention(x) * x + self.channel_attention(x) * x
        return self.activation(self.proj(attended) + self.shortcut(x))


@MODELS.register_module()
class UNetFormerBackbone(BaseModule):
    """ResNet encoder plus UNetFormer decoder returning main/aux features."""

    def __init__(
        self,
        backbone_name: str = 'resnet18.a1_in1k',
        pretrained: bool = True,
        decode_channels: int = 64,
        window_size: int = 8,
        drop_path_rate: float = 0.1,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        import timm

        self.encoder = timm.create_model(
            backbone_name,
            features_only=True,
            output_stride=32,
            out_indices=(1, 2, 3, 4),
            pretrained=pretrained,
        )
        channels = self.encoder.feature_info.channels()
        self.pre_conv = ConvBNAct(
            channels[-1], decode_channels, kernel_size=1, activation=False)
        self.block4 = GlobalLocalBlock(
            decode_channels, 8, window_size, drop_path_rate)
        self.fuse3 = WeightedFusion(channels[-2], decode_channels)
        self.block3 = GlobalLocalBlock(
            decode_channels, 8, window_size, drop_path_rate)
        self.fuse2 = WeightedFusion(channels[-3], decode_channels)
        self.block2 = GlobalLocalBlock(
            decode_channels, 8, window_size, drop_path_rate)
        self.refine1 = FeatureRefinementHead(channels[-4], decode_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        res1, res2, res3, res4 = self.encoder(x)
        x4 = self.block4(self.pre_conv(res4))
        x3 = self.block3(self.fuse3(x4, res3))
        x2 = self.block2(self.fuse2(x3, res2))
        x1 = self.refine1(x2, res1)
        auxiliary = F.interpolate(
            x4, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        auxiliary += F.interpolate(
            x3, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        auxiliary += x2
        return x1, auxiliary
