"""RGB-conditioned Gaussian primitives and differentiable orthographic splatting.

This is an unordered 2.5D feature representation, not a multi-view 3DGS
reconstruction. Heights and coverage are latent attributes, not measured depth
or calibrated confidence. All geometric paths remain differentiable.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .rpgv_v6_modules import _conv_block, _coordinates, _resize


def bounded_grid(size, maximum):
    scale = min(1.0, maximum / max(size))
    return tuple(max(1, round(value * scale)) for value in size)


def _geometry_dtype(value):
    return torch.float64 if value.dtype == torch.float64 else torch.float32


class GaussianSplatV8(nn.Module):
    """Normalized peak kernels; no depth sorting or alpha-over compositing."""

    def forward(self, primitives, size):
        reference = primitives['centers']
        with torch.autocast(device_type=reference.device.type, enabled=False):
            dtype = _geometry_dtype(reference)
            centers = reference.to(dtype)
            factor = primitives['cholesky'].to(dtype)
            amplitude = primitives['amplitude'].to(dtype)
            coordinates = _coordinates(*size, centers).flatten(2).transpose(1, 2)
            delta = coordinates[:, None] - centers[:, :, None]
            # Solve the triangular 2x2 system without per-pixel matrix inverses.
            dx = delta[..., 0] / factor[..., 0, 0, None]
            dy = (delta[..., 1] - factor[..., 1, 0, None] * dx) / factor[..., 1, 1, None]
            log_kernel = amplitude.clamp_min(torch.finfo(dtype).tiny).log()
            log_kernel = log_kernel - 0.5 * (dx.square() + dy.square())
            support = log_kernel.softmax(dim=1)
            log_mass = log_kernel.logsumexp(dim=1, keepdim=True)
            # Above this mass coverage is numerically one; avoid exp overflow.
            mass = log_mass.clamp_max(20.0).exp()
            coverage = -torch.expm1(-mass)
            heights = primitives['height'].to(dtype)
            height = torch.bmm(heights.transpose(1, 2), support)
            variance = (support * (heights - height).square()).sum(dim=1, keepdim=True)
            shape = (reference.shape[0], 1, *size)
            return dict(support=support, height=height.reshape(shape),
                        variance=variance.reshape(shape), coverage=coverage.reshape(shape),
                        size=tuple(size))

    @staticmethod
    def render(features, raster):
        with torch.autocast(device_type=features.device.type, enabled=False):
            support = raster['support']
            value = torch.bmm(features.to(support.dtype).transpose(1, 2), support)
            return value.reshape(features.shape[0], features.shape[-1], *raster['size'])


class GaussianPrimitivePredictorV8(nn.Module):
    def __init__(self, in_channels, channels=128, grid_size=8, max_size=64,
                 fixed_gaussians=False):
        super().__init__()
        if grid_size < 1 or max_size < 1:
            raise ValueError('Gaussian grid_size and max_size must be positive')
        self.grid_size = int(grid_size)
        self.max_size = int(max_size)
        self.fixed_gaussians = bool(fixed_gaussians)
        self.projections = nn.ModuleList([
            _conv_block(value, channels, 1) for value in in_channels])
        self.encoder = _conv_block(channels, channels)
        if not self.fixed_gaussians:
            self.center_head = nn.Linear(channels, 2)
            self.covariance_head = nn.Linear(channels, 3)
        self.amplitude_head = nn.Linear(channels, 1)
        self.height_head = nn.Linear(channels, 1)
        self.feature_head = nn.Linear(channels, channels)
        self.shape_projection = nn.Sequential(nn.Linear(7, channels), nn.GELU(approximate='tanh'),
                                              nn.Linear(channels, channels))
        self.node_norm = nn.LayerNorm(channels)
        self.splat = GaussianSplatV8()

    def forward(self, features):
        size = bounded_grid(features[0].shape[-2:], self.max_size)
        encoded = self.encoder(sum(_resize(layer(feature), size)
                                  for layer, feature in zip(self.projections, features))
                               / len(features))
        grid = (min(self.grid_size, size[0]), min(self.grid_size, size[1]))
        seeds = F.adaptive_avg_pool2d(encoded, grid).flatten(2).transpose(1, 2)
        with torch.autocast(device_type=encoded.device.type, enabled=False):
            seeds = seeds.to(_geometry_dtype(seeds))
            anchors = _coordinates(*grid, seeds).flatten(2).transpose(1, 2)
            step = seeds.new_tensor((2.0 / grid[1], 2.0 / grid[0]))
            if self.fixed_gaussians:
                centers = anchors.expand(seeds.shape[0], -1, -1)
                factor = torch.diag_embed((step * 0.8).expand(*seeds.shape[:2], 2))
            else:
                centers = anchors + self.center_head(seeds).tanh() * step
                parameters = self.covariance_head(seeds)
                diagonal = (F.softplus(parameters[..., :2]) + 1e-4) * step
                off_diagonal = parameters[..., 2] * step[1]
                zero = torch.zeros_like(off_diagonal)
                factor = torch.stack((diagonal[..., 0], zero, off_diagonal,
                                      diagonal[..., 1]), dim=-1).reshape(*seeds.shape[:2], 2, 2)
            amplitude = F.softplus(self.amplitude_head(seeds)) + 1e-4
            height = self.height_head(seeds).tanh()
            primitives = dict(centers=centers, cholesky=factor, amplitude=amplitude,
                              height=height)
            raster = self.splat(primitives, size)
            support = raster['support']
            pool = support / support.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            pooled = torch.bmm(pool, encoded.to(support.dtype).flatten(2).transpose(1, 2))
            shape = torch.cat((centers, factor[..., 0, 0, None], factor[..., 1, 0, None],
                               factor[..., 1, 1, None], height, amplitude.log()), dim=-1)
            nodes = self.node_norm(self.feature_head(seeds) + pooled + self.shape_projection(shape))
        return primitives, raster, nodes


class GaussianRelationLayerV8(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        if channels % heads:
            raise ValueError('Gaussian relation channels must be divisible by heads')
        self.heads = heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.edge_bias = nn.Sequential(nn.Linear(8, channels), nn.GELU(approximate='tanh'),
                                       nn.Linear(channels, heads))
        self.output = nn.Linear(channels, channels)
        self.ffn = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 2 * channels),
                                 nn.GELU(approximate='tanh'), nn.Linear(2 * channels, channels))

    def forward(self, nodes, edges):
        batch, count, channels = nodes.shape
        qkv = self.qkv(self.norm(nodes)).reshape(batch, count, 3, self.heads, channels // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(channels // self.heads)
        logits = logits + self.edge_bias(edges).permute(0, 3, 1, 2)
        message = torch.matmul(logits.softmax(dim=-1), value)
        message = message.transpose(1, 2).reshape(batch, count, channels)
        nodes = nodes + self.output(message)
        return nodes + self.ffn(nodes)


class GaussianRelationNetworkV8(nn.Module):
    def __init__(self, channels=128, heads=4, layers=2):
        super().__init__()
        if layers < 1:
            raise ValueError('relation network needs at least one layer')
        self.layers = nn.ModuleList([GaussianRelationLayerV8(channels, heads) for _ in range(layers)])

    def forward(self, nodes, primitives, raster):
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            nodes = nodes.to(_geometry_dtype(nodes))
            center = primitives['centers']
            factor = primitives['cholesky']
            covariance = factor @ factor.transpose(-1, -2)
            scale = covariance.diagonal(dim1=-2, dim2=-1).sqrt()
            correlation = covariance[..., 0, 1] / scale.prod(dim=-1).clamp_min(1e-8)
            delta = center[:, :, None] - center[:, None, :]
            log_scale = scale.log()
            ratio = log_scale[:, :, None] - log_scale[:, None, :]
            height = primitives['height']
            dz = height[:, :, None] - height[:, None, :]
            dc = correlation[:, :, None, None] - correlation[:, None, :, None]
            support = raster['support']
            unit = F.normalize(support, p=2, dim=-1, eps=1e-8)
            overlap = torch.bmm(unit, unit.transpose(1, 2)).unsqueeze(-1)
            distance = delta.square().sum(-1, keepdim=True)
            edges = torch.cat((delta, ratio, dz, dc, overlap, distance), dim=-1)
            for layer in self.layers:
                nodes = layer(nodes, edges)
        return nodes


class GaussianFusionV8(nn.Module):
    def __init__(self, decoder_channels, gaussian_channels):
        super().__init__()
        self.update = nn.Sequential(_conv_block(decoder_channels + gaussian_channels, decoder_channels, 1),
                                    _conv_block(decoder_channels, decoder_channels))

    def forward(self, feature, gaussian):
        return feature + self.update(torch.cat((feature, _resize(gaussian, feature.shape[-2:])), dim=1))
