"""Hierarchical cross-representation objectives; no labels construct the graph."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .rpgv_v7_modules import (
    RELATION_DIM, gather_nodes, local_compositions, masked_average, permute_composition,
)


def projection(in_channels, channels):
    return nn.Sequential(nn.Linear(in_channels, channels), nn.LayerNorm(channels),
                         nn.GELU(approximate='tanh'), nn.Linear(channels, channels))


class MeasuredPairContrastV7(nn.Module):
    """Balanced sigmoid pairs; similar nonmatches are unknown, not forced negatives."""
    def __init__(self, channels, temperature=0.1, similar_threshold=0.05):
        super().__init__()
        if not 0.01 <= temperature <= 1 or similar_threshold < 0:
            raise ValueError('invalid contrastive temperature or similarity threshold')
        self.rgb = projection(channels, channels)
        self.geometry = projection(channels, channels)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        self.similar_threshold = similar_threshold

    def forward(self, rgb, geometry, measured, valid):
        rgb = F.normalize(self.rgb(rgb.float()), dim=-1)
        geometry = F.normalize(self.geometry(geometry.float()), dim=-1)
        logits = self.logit_scale.clamp(0, math.log(100)).exp() * rgb @ geometry.T + self.logit_bias
        with torch.no_grad():
            distance = (measured[:, None] - measured[None, :]).abs().mean(-1)
            eye = torch.eye(valid.numel(), device=valid.device, dtype=torch.bool)
            negative = valid[:, None] & valid[None, :] & ~eye & (distance > self.similar_threshold)
        positive = masked_average(F.softplus(-logits.diagonal()), valid, 0)
        per_anchor = masked_average(F.softplus(logits), negative, -1)
        negative_term = masked_average(per_anchor, negative.any(-1), 0)
        return positive + negative_term, negative.sum().detach().float()


class HierarchicalContrastV7(nn.Module):
    def __init__(self, channels=128, node_weight=0.05, relation_weight=0.05,
                 composition_weight=0.05, relation_prediction_weight=0.25,
                 max_relations=128, composition_size=16, negatives=2,
                 margin=0.2, similar_threshold=0.05):
        super().__init__()
        weights = (node_weight, relation_weight, composition_weight, relation_prediction_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError('structure loss weights must be finite and nonnegative')
        if min(max_relations, composition_size, negatives) < 1 or margin < 0:
            raise ValueError('invalid relation/composition sampling configuration')
        self.node_weight, self.relation_weight, self.composition_weight = weights[:3]
        self.prediction_weight = relation_prediction_weight
        self.max_relations, self.composition_size = max_relations, composition_size
        self.negatives, self.margin = negatives, margin
        if node_weight:
            self.node_contrast = MeasuredPairContrastV7(channels, similar_threshold=similar_threshold)
        if relation_weight:
            self.rgb_relation = projection(channels * 2, channels)
            self.geometry_relation = projection(channels * 2 + RELATION_DIM, channels)
            self.relation_contrast = MeasuredPairContrastV7(channels, similar_threshold=similar_threshold)
            if relation_prediction_weight:
                self.relation_prediction = nn.Linear(channels, RELATION_DIM)
        if composition_weight:
            self.rgb_composition = projection(channels, channels)
            self.geometry_composition = projection(channels, channels)

    @property
    def enabled(self):
        return bool(self.node_weight or self.relation_weight or self.composition_weight)

    def forward(self, primitives, rgb_nodes, graph, adjacency):
        with torch.autocast(device_type=rgb_nodes.device.type, enabled=False):
            return self._forward(primitives, rgb_nodes.float(), graph, adjacency)

    def _forward(self, primitives, rgb, graph, adjacency):
        losses = {}
        valid = primitives['valid']
        geo = primitives['contrast_nodes'].float()
        if self.node_weight:
            raw, negative_count = self.node_contrast(
                rgb.flatten(0, 1), geo.flatten(0, 1),
                primitives['content'].flatten(0, 1), valid.flatten())
            losses.update(loss_node_v7=self.node_weight * raw, raw_node_v7=raw.detach(),
                          node_negative_count_v7=negative_count)
        if self.relation_weight:
            batch, count, channels = rgb.shape
            eye = torch.eye(count, device=rgb.device, dtype=torch.bool)
            eligible = adjacency & ~eye
            # Bounded samples from this microbatch; gradient accumulation is not a queue.
            priority = torch.rand(eligible.numel(), device=rgb.device)
            priority = priority.masked_fill(~eligible.flatten(), -1)
            selected = priority.topk(min(self.max_relations, priority.numel())).indices
            b = selected // (count * count)
            i, j = (selected // count) % count, selected % count
            observed = eligible[b, i, j]
            measured = primitives['edges'][b, i, j]
            rgb_pairs = self.rgb_relation(torch.cat((rgb[b, i], rgb[b, j]), -1))
            geometry_pairs = self.geometry_relation(torch.cat((geo[b, i], geo[b, j], measured), -1))
            contrast, negative_count = self.relation_contrast(rgb_pairs, geometry_pairs, measured, observed)
            prediction = contrast * 0
            if self.prediction_weight:
                error = F.smooth_l1_loss(self.relation_prediction(rgb_pairs), measured, reduction='none').mean(-1)
                prediction = masked_average(error, observed, 0)
            raw = contrast + self.prediction_weight * prediction
            losses.update(loss_relation_v7=self.relation_weight * raw,
                          raw_relation_v7=raw.detach(), relation_prediction_v7=prediction.detach(),
                          relation_count_v7=observed.sum().detach().float(),
                          relation_negative_count_v7=negative_count)
        if self.composition_weight:
            local = local_compositions(primitives, adjacency, self.composition_size)
            b, n, k, channels = local['nodes'].shape
            rgb_local = gather_nodes(rgb, local['indices'])
            rgb_pooled = masked_average(rgb_local, local['valid'].unsqueeze(-1), -2)
            query = F.normalize(self.rgb_composition(rgb_pooled), dim=-1)

            def encode(candidate):
                nodes, _ = graph(candidate['nodes'].reshape(b * n, k, channels),
                                 candidate['content'].reshape(b * n, k, -1),
                                 candidate['positions'].reshape(b * n, k, 2),
                                 candidate['contact'].reshape(b * n, k, k),
                                 candidate['valid'].reshape(b * n, k))
                nodes = nodes.reshape(b, n, k, channels)
                pooled = masked_average(nodes, candidate['valid'].unsqueeze(-1), -2)
                return F.normalize(self.geometry_composition(pooled), dim=-1)

            positive = (query * encode(local)).sum(-1)
            terms, masks, gaps = [], [], []
            for _ in range(self.negatives):
                candidate, changed = permute_composition(local)
                negative = (query * encode(candidate)).sum(-1)
                gap = positive - negative
                terms.append(F.softplus(self.margin - gap))
                masks.append(changed & valid)
                gaps.append(gap)
            observed = torch.stack(masks)
            # Average candidates per composition, then compositions (not number of edges).
            error = masked_average(torch.stack(terms), observed, 0)
            raw = masked_average(error, observed.any(0), (0, 1))
            gap = masked_average(torch.stack(gaps).detach(), observed, (0, 1, 2))
            losses.update(loss_composition_v7=self.composition_weight * raw,
                          raw_composition_v7=raw.detach(), composition_gap_v7=gap,
                          composition_count_v7=observed.any(0).sum().detach().float(),
                          changed_candidate_count_v7=observed.sum().detach().float())
        losses['node_count_v7'] = valid.sum().detach().float()
        return losses
