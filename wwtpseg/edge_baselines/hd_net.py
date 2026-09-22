# SPDX-License-Identifier: GPL-3.0-only
"""HD-Net architecture adapted from danfenghong/ISPRS_HD-Net (186d656).

This implementation follows its two high-resolution stages, four dilated
three-branch stages, flow-based body/boundary decoupling, and six deeply
supervised segmentation and boundary outputs. It is kept in this dedicated
comparison package because the upstream implementation is GPL-3.0.
Original work copyright (C) 2024 Danfeng Hong and contributors.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _conv_bn(in_channels: int, out_channels: int, kernel: int = 3,
             stride: int = 1, activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel, stride,
                  padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class _Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            _conv_bn(in_channels, 64, 1),
            _conv_bn(64, 64),
            _conv_bn(64, out_channels, 1, activation=False),
        )
        self.skip = (_conv_bn(in_channels, out_channels, 1, activation=False)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.blocks(x) + self.skip(x), inplace=False)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.first = nn.Conv2d(channels, channels, 3,
                               padding=dilation, dilation=dilation, bias=False)
        self.first_bn = nn.BatchNorm2d(channels)
        self.second = _conv_bn(channels, channels, activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.first_bn(self.first(x)), inplace=True)
        return F.relu(self.second(residual) + x, inplace=False)


class _DecoupledStage(nn.Module):
    def __init__(self, channels: int, branches: int, dilation: int = 0) -> None:
        super().__init__()
        self.branches = branches
        self.transforms = nn.ModuleList()
        for index in range(branches):
            width = channels * 2 ** index
            dilations = ([1] * 4 if dilation == 0 else
                         [dilation, 2 * dilation, 4 * dilation])
            self.transforms.append(nn.Sequential(*[
                _ResidualBlock(width, amount) for amount in dilations]))
        self.to_high = nn.ModuleList([nn.Identity()] + [
            _conv_bn(channels * 2 ** i, channels, 1, activation=False)
            for i in range(1, branches)])
        self.to_low = nn.ModuleList([
            nn.Sequential(*(
                [_conv_bn(channels, channels, stride=2)] * (index - 1) +
                [_conv_bn(channels, channels * 2 ** index,
                          stride=2, activation=False)]))
            for index in range(1, branches)])
        self.flow = nn.Conv2d(channels * branches, 2, 3, padding=1,
                              bias=False)
        self.edge_fusion = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.body_fusion = nn.Conv2d(channels * branches, channels, 1,
                                     bias=False)
        self.edge_out = nn.Conv2d(channels, 1, 1, bias=False)
        self.body_out = nn.Conv2d(channels, 1, 1, bias=False)

    def forward(self, features: list[torch.Tensor], fine: torch.Tensor
                ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        features = [block(x) for block, x in zip(self.transforms, features)]
        height, width = features[0].shape[-2:]
        high = features[0]
        abstract = [F.interpolate(
            project(value), size=(height, width), mode='bilinear',
            align_corners=False)
            for project, value in zip(self.to_high[1:], features[1:])]
        flow = self.flow(torch.cat([high, *abstract], dim=1))
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=high.device, dtype=high.dtype),
            torch.linspace(-1, 1, width, device=high.device, dtype=high.dtype),
            indexing='ij')
        grid = torch.stack((x, y), dim=-1)[None] + (
            flow.permute(0, 2, 3, 1) /
            flow.new_tensor((width, height)).view(1, 1, 1, 2))
        body = F.grid_sample(high, grid, align_corners=False)
        edge = self.edge_fusion(torch.cat((high - body, fine), dim=1))
        body = self.body_fusion(torch.cat((body, *abstract), dim=1))
        edge_logit = self.edge_out(edge)
        body_logit = self.body_out(body)
        edge_gate = edge_logit.sigmoid().detach()
        body_gate = body_logit.sigmoid().detach()
        combined = edge * edge_gate * (1 - body_gate) + \
            body * body_gate * (1 - edge_gate)
        top = F.relu(high + combined, inplace=False)
        out = [top]
        for index in range(1, self.branches):
            projected = self.to_low[index - 1](top)
            lower = features[index] + projected
            out.append(F.relu(lower, inplace=False))
        return out, edge, edge_logit


def _prediction_head(channels: int, in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, channels, 1),
        nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        nn.Conv2d(channels, 1, 1))


class HDNet(nn.Module):
    """Six-stage high-resolution body/boundary decomposition network."""

    def __init__(self, base_channel: int = 48) -> None:
        super().__init__()
        c = base_channel
        self.stem = nn.Sequential(
            _conv_bn(3, 64), _conv_bn(64, 64, stride=2),
            _conv_bn(64, 64),
            *[_Bottleneck(64 if i == 0 else 256) for i in range(4)])
        self.fine = nn.Conv2d(256, c, 1, bias=False)
        self.transition_high = _conv_bn(256, c)
        self.transition_low = _conv_bn(256, 2 * c, stride=2)
        self.transition_third = _conv_bn(2 * c, 4 * c, stride=2)
        self.stages = nn.ModuleList([
            _DecoupledStage(c, 2), _DecoupledStage(c, 2),
            *[_DecoupledStage(c, 3, dilation=2 ** (i + 1))
              for i in range(4)],
        ])
        self.seg_heads = nn.ModuleList([
            _prediction_head(c, 3 * c),
            _prediction_head(c, 3 * c),
            *[_prediction_head(c, 7 * c) for _ in range(4)],
        ])
        self.final_seg = nn.Conv2d(6, 1, 1)
        self.final_boundary = nn.Conv2d(6, 1, 1)

    @staticmethod
    def _concat(features: list[torch.Tensor]) -> torch.Tensor:
        size = features[0].shape[-2:]
        return torch.cat([features[0], *[
            F.interpolate(x, size=size, mode='bilinear', align_corners=True)
            for x in features[1:]]], dim=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | list]:
        output_size = image.shape[-2:]
        stem = self.stem(image)
        fine = self.fine(stem)
        features = [self.transition_high(stem), self.transition_low(stem)]
        segments = []
        edges = []
        for index, (stage, head) in enumerate(zip(self.stages,
                                                   self.seg_heads)):
            if index == 2:
                features.append(self.transition_third(features[-1]))
            features, fine, edge = stage(features, fine)
            joined = self._concat(features)
            if index == 5:
                joined = F.interpolate(joined, size=output_size,
                                       mode='bilinear', align_corners=False)
            segments.append(head(joined))
            edges.append(edge)
        # The upstream sixth auxiliary output and both fused predictions are
        # full resolution; the first five auxiliaries remain at half resolution.
        edges[-1] = F.interpolate(edges[-1], size=output_size,
                                   mode='bilinear', align_corners=False)
        full_segments = [F.interpolate(x, size=output_size, mode='bilinear',
                                       align_corners=False) for x in segments]
        full_edges = [F.interpolate(x, size=output_size, mode='bilinear',
                                    align_corners=False) for x in edges]
        return dict(final=self.final_seg(torch.cat(full_segments, dim=1)),
                    boundary=self.final_boundary(torch.cat(full_edges, dim=1)),
                    segments=segments, edges=edges)
