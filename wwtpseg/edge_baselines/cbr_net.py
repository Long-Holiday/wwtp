"""CBR-Net architecture adapted from HaonanGuo/CBRNet (commit 5d521ce).

The model retains the VGG16-BN encoder, four progressive U-Net refinements,
multi-scale edge heads and the eight-direction boundary correction. Data
handling and optimization are supplied by the local MMSeg experiment.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG16_BN_Weights, vgg16_bn


class _DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, deep: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        deep = F.interpolate(deep, size=skip.shape[-2:], mode='bilinear',
                             align_corners=True)
        return self.conv(torch.cat((skip, deep), dim=1))


def _head(in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 1), nn.BatchNorm2d(64),
        nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))


class CBRNet(nn.Module):
    """Full-resolution foreground logit plus the original auxiliary heads."""

    # The channel order matches the fixed shift bank in the upstream model.
    DIRECTIONS = ((0, -1), (-1, -1), (-1, 0), (-1, 1),
                  (0, 1), (1, 1), (1, 0), (1, -1))

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        features = vgg16_bn(weights=weights).features
        self.encoder = nn.ModuleList(
            features[start:end] for start, end in
            ((0, 5), (5, 12), (12, 22), (22, 32), (32, 42)))
        self.coarse_head = _head(512)
        self.ups = nn.ModuleList((_Up(1024, 256), _Up(512, 128),
                                  _Up(256, 64), _Up(128, 64)))
        self.seg_heads = nn.ModuleList(
            _head(channels + 1) for channels in (256, 128, 64, 64))
        self.edge_heads = nn.ModuleList(
            _head(channels) for channels in (256, 128, 64, 64))
        self.direction_head = nn.Sequential(
            nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.Conv2d(64, 8, 1))
        shifts = torch.zeros(8, 1, 3, 3)
        for index, (dy, dx) in enumerate(self.DIRECTIONS):
            shifts[index, 0, 1 + dy, 1 + dx] = 1
        self.register_buffer('shift_kernel', shifts, persistent=False)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | list]:
        features = []
        for block in self.encoder:
            image = block(image)
            features.append(image)
        previous = self.coarse_head(features[-1])
        segmentations = [previous]
        edges = []
        for up, seg_head, edge_head, skip in zip(
                self.ups, self.seg_heads, self.edge_heads,
                reversed(features[:-1])):
            image = up(image, skip)
            edges.append(edge_head(image))
            guidance = F.interpolate(previous, size=image.shape[-2:],
                                     mode='bilinear', align_corners=False)
            previous = seg_head(torch.cat((image, guidance), dim=1))
            segmentations.append(previous)
        direction = self.direction_head(image)
        shifted = F.conv2d(previous, self.shift_kernel.to(previous.dtype),
                           padding=1)
        corrected = (shifted * direction.softmax(dim=1)).sum(
            dim=1, keepdim=True)
        boundary_gate = (edges[-1].sigmoid().detach() > 0.5).to(previous.dtype)
        refined = boundary_gate * corrected + (1 - boundary_gate) * previous
        return dict(final=refined, segments=segmentations,
                    edges=edges, direction=direction)
