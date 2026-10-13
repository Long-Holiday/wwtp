"""Content-adaptive Gaussian supports and relation messages for RPGV v8.1."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .rpgv_v6_modules import _conv_block, _coordinates, _resize
from .rpgv_v8_modules import (GaussianPrimitivePredictorV8, GaussianRelationLayerV8,
                              GaussianRelationNetworkV8, GaussianSplatV8,
                              _geometry_dtype, bounded_grid)


class ContentGaussianSplatV81(GaussianSplatV8):
    def forward(self, primitives, size, content_logits=None):
        if content_logits is None:
            return super().forward(primitives, size)
        reference = primitives['centers']
        with torch.autocast(device_type=reference.device.type, enabled=False):
            dtype = _geometry_dtype(reference)
            centers = reference.to(dtype)
            factor = primitives['cholesky'].to(dtype)
            amplitude = primitives['amplitude'].to(dtype)
            coordinates = _coordinates(*size, centers).flatten(2).transpose(1, 2)
            delta = coordinates[:, None] - centers[:, :, None]
            dx = delta[..., 0] / factor[..., 0, 0, None]
            dy = (delta[..., 1] - factor[..., 1, 0, None] * dx) / factor[..., 1, 1, None]
            log_kernel = amplitude.clamp_min(torch.finfo(dtype).tiny).log()
            log_kernel = log_kernel - 0.5 * (dx.square() + dy.square())
            support = (log_kernel + content_logits.to(dtype)).softmax(dim=1)
            # Coverage measures Gaussian mass, not content similarity or confidence.
            mass = log_kernel.logsumexp(dim=1, keepdim=True).clamp_max(20.0).exp()
            coverage = -torch.expm1(-mass)
            heights = primitives['height'].to(dtype)
            height = torch.bmm(heights.transpose(1, 2), support)
            variance = (support * (heights - height).square()).sum(dim=1, keepdim=True)
            shape = (reference.shape[0], 1, *size)
            return dict(support=support, height=height.reshape(shape),
                        variance=variance.reshape(shape), coverage=coverage.reshape(shape),
                        size=tuple(size))


class GaussianPrimitivePredictorV81(GaussianPrimitivePredictorV8):
    def __init__(self, in_channels, channels=128, content_aware=True, content_scale=2.0,
                 **kwargs):
        super().__init__(in_channels, channels=channels, **kwargs)
        if not math.isfinite(content_scale) or content_scale <= 0:
            raise ValueError('content_scale must be finite and positive')
        self.content_aware = bool(content_aware)
        self.content_scale = float(content_scale)
        self.splat = ContentGaussianSplatV81()
        if self.content_aware:
            self.content_query = nn.Linear(channels, channels, bias=False)
            self.content_key = nn.Linear(channels, channels, bias=False)

    def forward(self, features):
        if not self.content_aware:
            return super().forward(features)
        size = bounded_grid(features[0].shape[-2:], self.max_size)
        encoded = self.encoder(sum(_resize(layer(feature), size)
                                  for layer, feature in zip(self.projections, features))
                               / len(features))
        levels = [(min(value, size[0]), min(value, size[1])) for value in self.grid_sizes]
        seeds = torch.cat([F.adaptive_avg_pool2d(encoded, grid).flatten(2).transpose(1, 2)
                           for grid in levels], dim=1)
        with torch.autocast(device_type=encoded.device.type, enabled=False):
            seeds = seeds.to(_geometry_dtype(seeds))
            anchors = torch.cat([_coordinates(*grid, seeds).flatten(2).transpose(1, 2)
                                 for grid in levels], dim=1)
            step = torch.cat([seeds.new_tensor((2.0 / grid[1], 2.0 / grid[0])).expand(
                grid[0] * grid[1], 2) for grid in levels], dim=0).unsqueeze(0)
            if self.fixed_gaussians:
                centers = anchors.expand(seeds.shape[0], -1, -1)
                factor = torch.diag_embed(step.expand(*seeds.shape[:2], 2) * 0.8)
            else:
                centers = anchors + self.center_head(seeds).tanh() * step * self.center_log_scale.exp()
                parameters = self.covariance_head(seeds)
                diagonal = (F.softplus(parameters[..., :2]) + 1e-4) * step
                off_diagonal = parameters[..., 2] * step[..., 1]
                factor = torch.stack((diagonal[..., 0], torch.zeros_like(off_diagonal),
                                      off_diagonal, diagonal[..., 1]), dim=-1).reshape(
                                          *seeds.shape[:2], 2, 2)
            amplitude = F.softplus(self.amplitude_head(seeds)) + 1e-4
            height = self.height_head(seeds).tanh()
            primitives = dict(centers=centers, cholesky=factor, amplitude=amplitude, height=height)
            pixels = encoded.to(seeds.dtype).flatten(2).transpose(1, 2)
            query = F.normalize(self.content_query(pixels), dim=-1, eps=1e-8)
            key = F.normalize(self.content_key(seeds), dim=-1, eps=1e-8)
            content_logits = self.content_scale * torch.bmm(key, query.transpose(1, 2))
            raster = self.splat(primitives, size, content_logits)
            support = raster['support']
            pool = support / support.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            pooled = torch.bmm(pool, pixels)
            shape = torch.cat((centers, factor[..., 0, 0, None], factor[..., 1, 0, None],
                               factor[..., 1, 1, None], height, amplitude.log()), dim=-1)
            nodes = self.node_norm(self.feature_head(seeds) + pooled + self.shape_projection(shape))
        return primitives, raster, nodes


class GaussianRelationLayerV81(GaussianRelationLayerV8):
    def __init__(self, channels, heads, use_contact=True, use_edge_value=True):
        super().__init__(channels, heads, use_contact=use_contact)
        self.use_edge_value = bool(use_edge_value)
        if self.use_edge_value:
            self.edge_value = nn.Linear(self.edge_bias[0].in_features, channels, bias=False)

    def forward(self, nodes, edges):
        if not self.use_edge_value:
            return super().forward(nodes, edges)
        batch, count, channels = nodes.shape
        width = channels // self.heads
        qkv = self.qkv(self.norm(nodes)).reshape(batch, count, 3, self.heads, width)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(width)
        attention = (logits + self.edge_bias(edges).permute(0, 3, 1, 2)).softmax(dim=-1)
        message = torch.matmul(attention, value)
        # Aggregate the small descriptor before projecting; avoid B*N*N*C edge values.
        relation = torch.einsum('bhij,bijd->bhid', attention, edges)
        weights = self.edge_value.weight.reshape(self.heads, width, edges.shape[-1])
        message = message + torch.einsum('bhid,hcd->bhic', relation, weights)
        message = message.transpose(1, 2).reshape(batch, count, channels)
        nodes = nodes + self.output(message)
        return nodes + self.ffn(nodes)


class GaussianRelationNetworkV81(GaussianRelationNetworkV8):
    def __init__(self, channels=128, heads=4, layers=2, use_contact=True, use_edge_value=True):
        nn.Module.__init__(self)
        if layers < 1:
            raise ValueError('relation network needs at least one layer')
        self.use_contact = bool(use_contact)
        self.layers = nn.ModuleList([
            GaussianRelationLayerV81(channels, heads, self.use_contact, use_edge_value)
            for _ in range(layers)])


class LocalGaussianFusionV81(nn.Module):
    def __init__(self, rgb_channels, decoder_channels, gaussian_channels):
        super().__init__()
        self.rgb_projection = _conv_block(rgb_channels, decoder_channels, 1)
        self.gaussian_projection = _conv_block(gaussian_channels, decoder_channels, 1)
        self.update = nn.Sequential(_conv_block(3 * decoder_channels, decoder_channels, 1),
                                    _conv_block(decoder_channels, decoder_channels))

    def forward(self, feature, gaussian, rgb):
        detail = _resize(self.rgb_projection(rgb), feature.shape[-2:])
        context = _resize(self.gaussian_projection(gaussian), feature.shape[-2:])
        return feature + self.update(torch.cat((feature, detail, context), dim=1))
