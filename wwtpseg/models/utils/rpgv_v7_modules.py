"""Geometry primitives and edge-conditioned reasoning for RPGV v7."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .rpgv_v6_modules import _conv_block, _coordinates, _resize, _supported_derivatives


# Content: log area, xx/yy/xy central moments, mean/standard deviation of depth.
CONTENT_DIM = 6
# Relations: distance, log area ratio, relative depth, contact, dx, dy.
RELATION_DIM = 6


def masked_average(value, mask, dim):
    weight = mask.to(value.dtype)
    return (value * weight).sum(dim) / weight.sum(dim).clamp_min(1e-12)


def gather_nodes(nodes, indices):
    """B,N,C nodes -> B,...,C without expanding the source over all queries."""
    batch = torch.arange(nodes.shape[0], device=nodes.device)
    batch = batch.view(nodes.shape[0], *([1] * (indices.ndim - 1)))
    return nodes[batch, indices]


def relation_descriptors(content, positions, contact):
    delta = positions.unsqueeze(-3) - positions.unsqueeze(-2)  # position_j - position_i
    distance = delta.square().sum(-1).clamp_min(1e-12).sqrt()
    area = content[..., 0].unsqueeze(-2) - content[..., 0].unsqueeze(-1)
    depth = content[..., 4].unsqueeze(-2) - content[..., 4].unsqueeze(-1)
    return torch.cat((distance.unsqueeze(-1), area.unsqueeze(-1), depth.unsqueeze(-1),
                      contact.unsqueeze(-1), delta), dim=-1)


def spatial_neighbors(positions, contact, valid, neighbors):
    """Bounded nearest/contact graph, including self only for supported nodes."""
    count = positions.shape[-2]
    distance = (positions.unsqueeze(-3) - positions.unsqueeze(-2)).square().sum(-1)
    pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    eye = torch.eye(count, dtype=torch.bool, device=positions.device)
    # Contact is a tie-breaking spatial prior, not a learned sample selector.
    score = (distance - 0.1 * contact).masked_fill(~pair_valid | eye, float('inf'))
    k = min(neighbors, max(0, count - 1))
    adjacency = torch.zeros_like(pair_valid)
    if k:
        selected = score.topk(k, dim=-1, largest=False).indices
        adjacency.scatter_(-1, selected, True)
    return (adjacency | eye) & pair_valid


class GeometryPrimitivesV7(nn.Module):
    """Joint low/high geometry partitions; flat observations remain eligible."""

    def __init__(self, channels=128, grid_size=8, max_size=64, iterations=2,
                 temperature=0.2, spatial_weight=2.0, fixed_grid=False):
        super().__init__()
        if min(channels, grid_size, max_size, iterations) < 1:
            raise ValueError('primitive dimensions and iterations must be positive')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('primitive temperature must be positive')
        if not math.isfinite(spatial_weight) or spatial_weight < 0:
            raise ValueError('spatial weight must be nonnegative')
        self.encoder = nn.Sequential(_conv_block(5, channels), _conv_block(channels, channels))
        self.grid_size, self.max_size, self.iterations = grid_size, max_size, iterations
        self.temperature, self.spatial_weight = temperature, spatial_weight
        self.fixed_grid = bool(fixed_grid)

    @staticmethod
    def pool(features, pooling):
        features = F.adaptive_avg_pool2d(features.float(), pooling.shape[-2:])
        return torch.bmm(pooling.flatten(2), features.flatten(2).transpose(1, 2))

    def forward(self, depth, low, high, quality, measured_depth=None):
        with torch.autocast(device_type=depth.device.type, enabled=False):
            quality = quality.detach().float().clamp(0, 1)
            depth = torch.where(quality > 0, depth.float(), 0.0)
            size = tuple(min(s, self.max_size) for s in depth.shape[-2:])
            q = F.adaptive_avg_pool2d(quality, size)
            reduced_depth = F.adaptive_avg_pool2d(depth * quality, size) / q.clamp_min(1e-12)
            dx, dy = _supported_derivatives(reduced_depth, q)
            # Only the existing v6 detail path needs full F4 resolution. Encode the
            # structure grid directly instead of allocating 128-channel F4 maps.
            geo = self.encoder(torch.cat((reduced_depth,
                F.adaptive_avg_pool2d(low.float(), size),
                F.adaptive_avg_pool2d(high.float(), size), dx, dy), 1))
            # Zero Q values are absent, not geometric boundaries or observations.
            support = (q > 0).float()
            grid = tuple(min(s, self.grid_size) for s in size)
            coords = _coordinates(*size, geo).flatten(2).expand(geo.shape[0], -1, -1)
            positions = _coordinates(*grid, geo).flatten(2).expand(geo.shape[0], -1, -1)
            seed_q = F.adaptive_avg_pool2d(q, grid).flatten(2)
            centers = F.adaptive_avg_pool2d(geo * q, grid).flatten(2) / seed_q.clamp_min(1e-12)
            features = geo.flatten(2)
            if self.fixed_grid:
                y = torch.arange(size[0], device=geo.device) * grid[0] // size[0]
                x = torch.arange(size[1], device=geo.device) * grid[1] // size[1]
                labels = (y[:, None] * grid[1] + x[None, :]).flatten()
                assignment = F.one_hot(labels, grid[0] * grid[1]).T.float()
                assignment = assignment.unsqueeze(0).expand(geo.shape[0], -1, -1)
            else:
                for _ in range(self.iterations):
                    similarity = torch.bmm(F.normalize(centers, dim=1).transpose(1, 2),
                                           F.normalize(features, dim=1))
                    distance = (positions.unsqueeze(-1) - coords.unsqueeze(2)).square().sum(1)
                    assignment = ((similarity - self.spatial_weight * distance)
                                  / self.temperature).softmax(1)
                    weights = assignment * q.flatten(2)
                    pooling = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
                    centers = torch.bmm(features, pooling.transpose(1, 2))
                    positions = torch.bmm(coords, pooling.transpose(1, 2))
            weights = assignment * q.flatten(2)
            mass = weights.sum(-1)
            pooling = weights / mass.unsqueeze(-1).clamp_min(1e-12)
            nodes = torch.bmm(pooling, features.transpose(1, 2))
            contrast_nodes = torch.bmm(pooling.detach(), features.transpose(1, 2))
            valid = (mass > 0).detach()
            # Targets measure the unrectified observation by default at the call site.
            observation = depth if measured_depth is None else measured_depth.float()
            observed = torch.where(quality > 0, observation, 0.0)
            observed = F.adaptive_avg_pool2d(observed * quality, size) / q.clamp_min(1e-12)
            with torch.no_grad():
                spatial = assignment.detach() * support.flatten(2)
                area = spatial.sum(-1)
                spatial_pool = spatial / area.unsqueeze(-1).clamp_min(1e-12)
                pos = torch.bmm(spatial_pool, coords.transpose(1, 2))
                xx = torch.bmm(spatial_pool, coords.square().transpose(1, 2)) - pos.square()
                xy = torch.bmm(spatial_pool, (coords[:, :1] * coords[:, 1:]).transpose(1, 2))
                xy = xy - pos[..., :1] * pos[..., 1:]
                mean = torch.bmm(pooling.detach(), observed.flatten(2).transpose(1, 2))
                variance = torch.bmm(pooling.detach(), observed.square().flatten(2).transpose(1, 2))
                std = (variance - mean.square()).clamp_min(0).sqrt()
                log_area = (area / support.sum((-2, -1)).clamp_min(1)).clamp_min(1e-12).log()
                content = torch.cat((log_area.unsqueeze(-1), xx.clamp_min(0), xy, mean, std), -1)
                maps = spatial.reshape(geo.shape[0], -1, *size)
                contact = torch.zeros(geo.shape[0], area.shape[1], area.shape[1], device=geo.device)
                for dim in (-2, -1):
                    a, b = maps.narrow(dim, 0, size[dim] - 1), maps.narrow(dim, 1, size[dim] - 1)
                    cross = torch.bmm(a.flatten(2), b.flatten(2).transpose(1, 2))
                    contact += cross + cross.transpose(1, 2)
                contact = contact / (area.unsqueeze(-1) * area.unsqueeze(-2)).clamp_min(1e-12).sqrt()
                content = torch.where(valid.unsqueeze(-1), content, 0.0)
                edges = relation_descriptors(content, pos, contact)
            return dict(nodes=nodes, contrast_nodes=contrast_nodes, content=content,
                        positions=pos, contact=contact, edges=edges, valid=valid,
                        assignment=assignment.reshape(geo.shape[0], -1, *size),
                        pooling=pooling.detach().reshape(geo.shape[0], -1, *size),
                        quality=q, support=support, region_quality=mass / area.clamp_min(1e-12))


class RelationLayerV7(nn.Module):
    def __init__(self, channels=128, heads=4):
        super().__init__()
        if channels % heads or min(channels, heads) < 1:
            raise ValueError('relation channels must be divisible by positive head count')
        self.heads, self.channels = heads, channels
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.edge_bias = nn.Linear(RELATION_DIM, heads)
        self.edge_value = nn.Linear(RELATION_DIM, channels)
        self.output = nn.Linear(channels, channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(approximate='tanh'),
                                 nn.Linear(channels * 2, channels))

    def forward(self, nodes, edges, adjacency, valid):
        batch, count, channels = nodes.shape
        q, k, v = self.qkv(self.norm(nodes)).reshape(
            batch, count, 3, self.heads, channels // self.heads).unbind(2)
        score = torch.einsum('bihd,bjhd->bhij', q, k) / math.sqrt(channels // self.heads)
        score = score + self.edge_bias(edges).permute(0, 3, 1, 2)
        attention = score.masked_fill(~adjacency[:, None], -1e4).softmax(-1)
        attention = attention * adjacency[:, None]
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
        value = self.edge_value(edges).reshape(batch, count, count, self.heads, -1)
        message = torch.einsum('bhij,bjhd->bihd', attention, v)
        message = message + torch.einsum('bhij,bijhd->bihd', attention, value)
        result = nodes + self.output(message.reshape(batch, count, channels))
        result = result + self.ffn(self.ffn_norm(result))
        return result * valid.unsqueeze(-1)


class GeometryRelationNetworkV7(nn.Module):
    def __init__(self, channels=128, heads=4, layers=2, neighbors=8, use_edges=True):
        super().__init__()
        if layers < 1 or neighbors < 1:
            raise ValueError('relation layers and neighbors must be positive')
        self.neighbors, self.use_edges = neighbors, bool(use_edges)
        self.content_projection = nn.Linear(CONTENT_DIM, channels)
        if use_edges:
            self.layers = nn.ModuleList([RelationLayerV7(channels, heads) for _ in range(layers)])
        else:
            self.layers = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels * 2),
                              nn.GELU(approximate='tanh'), nn.Linear(channels * 2, channels))
                for _ in range(layers)])

    def forward(self, nodes, content, positions, contact, valid):
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            nodes = nodes.float() + self.content_projection(content.float())
            nodes = nodes * valid.unsqueeze(-1)
            edges = relation_descriptors(content.float(), positions.float(), contact.float())
            adjacency = spatial_neighbors(positions, contact, valid, self.neighbors)
            for layer in self.layers:
                if self.use_edges:
                    nodes = layer(nodes, edges, adjacency, valid)
                else:
                    nodes = (nodes + layer(nodes)) * valid.unsqueeze(-1)
            return nodes, adjacency


def local_compositions(primitives, adjacency, size=16):
    """Select bounded two-hop neighborhoods; unsupported padding remains masked."""
    with torch.no_grad():
        reachable = adjacency | (torch.bmm(adjacency.float(), adjacency.float()) > 0)
        distance = (primitives['positions'][:, :, None] - primitives['positions'][:, None]).square().sum(-1)
        # Always retain the anchor even for coincident centroids.
        eye = torch.eye(distance.shape[-1], device=distance.device, dtype=torch.bool)
        distance = distance.masked_fill(eye, -1)
        indices = distance.masked_fill(~reachable, float('inf')).topk(
            min(size, distance.shape[-1]), largest=False, dim=-1).indices
        valid = reachable.gather(-1, indices)
        batch = torch.arange(indices.shape[0], device=indices.device)[:, None, None, None]
        contact = primitives['contact'][batch, indices[..., :, None], indices[..., None, :]]
    return dict(indices=indices, valid=valid,
                nodes=gather_nodes(primitives['contrast_nodes'], indices),
                content=gather_nodes(primitives['content'], indices),
                positions=gather_nodes(primitives['positions'], indices), contact=contact)


def permute_composition(local, permutation=None, minimum_change=1e-4):
    """Reassign content to positions, not graph node relabeling; no GT involved."""
    nodes, content, valid = local['nodes'], local['content'], local['valid']
    count = nodes.shape[-2]
    with torch.no_grad():
        if permutation is None:
            random = torch.rand(valid.shape, device=nodes.device).masked_fill(~valid, 2.0)
            shuffled = random.argsort(-1)
            ranks = (valid.long().cumsum(-1) - 1).clamp_min(0)
            selected = shuffled.gather(-1, ranks)
            identity = torch.arange(count, device=nodes.device).expand_as(selected)
            permutation = torch.where(valid, selected, identity)
        if permutation.shape != valid.shape:
            raise ValueError('composition permutation has wrong shape')
    moved_nodes = nodes.gather(-2, permutation.unsqueeze(-1).expand_as(nodes))
    moved_content = content.gather(-2, permutation.unsqueeze(-1).expand_as(content))
    with torch.no_grad():
        old = relation_descriptors(content, local['positions'], local['contact'])
        new = relation_descriptors(moved_content, local['positions'], local['contact'])
        pairs = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        difference = ((new - old).abs().mean(-1) * pairs).sum((-2, -1))
        difference = difference / pairs.sum((-2, -1)).clamp_min(1)
        changed = (difference > minimum_change) & (valid.sum(-1) > 1)
    return dict(local, nodes=moved_nodes, content=moved_content), changed
