"""Frequency and region-alignment blocks for RPGV v6.

Tanh GELU avoids PyTorch 2.1 CPU backward corruption when pooling/attention
produces a gradient with a different memory layout from the activation input.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .rpgv_modules import BoundedModulatedDeformConv2d


def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


def _bounded_size(x: torch.Tensor, maximum: int) -> tuple[int, int]:
    return min(x.shape[-2], maximum), min(x.shape[-1], maximum)


def _coordinates(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
    y = (torch.arange(height, device=reference.device, dtype=reference.dtype) + 0.5)
    x = (torch.arange(width, device=reference.device, dtype=reference.dtype) + 0.5)
    yy, xx = torch.meshgrid(y * (2.0 / height) - 1.0,
                            x * (2.0 / width) - 1.0, indexing='ij')
    return torch.stack((xx, yy), dim=0).unsqueeze(0)


def _conv_block(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    groups = min(8, max(1, out_channels // 2))
    while out_channels % groups:
        groups -= 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.GELU(approximate='tanh'),
    )


def _supported_derivatives(depth: torch.Tensor, quality: torch.Tensor):
    """Missing neighbours do not create a depth jump into a zero-filled hole."""
    padded = F.pad(depth, (1, 1, 1, 1), mode='replicate')
    observed = F.pad((quality > 0).to(depth.dtype), (1, 1, 1, 1), mode='replicate')
    dx = 0.5 * ((padded[..., 1:-1, 2:] - depth) * observed[..., 1:-1, 2:]
                + (depth - padded[..., 1:-1, :-2]) * observed[..., 1:-1, :-2])
    dy = 0.5 * ((padded[..., 2:, 1:-1] - depth) * observed[..., 2:, 1:-1]
                + (depth - padded[..., :-2, 1:-1]) * observed[..., :-2, 1:-1])
    return dx * (quality > 0), dy * (quality > 0)


class DeformableDepthRectifierV6(nn.Module):
    """Geometry-only, bounded depth correction at the first backbone resolution."""

    def __init__(self, channels: int = 16, max_offset: float = 2.0,
                 max_correction: float = 0.1) -> None:
        super().__init__()
        if channels < 2 or not math.isfinite(max_offset) or max_offset <= 0:
            raise ValueError('invalid rectifier channels or offset bound')
        if not math.isfinite(max_correction) or not 0 < max_correction <= 1:
            raise ValueError('max_correction must be finite and in (0, 1]')
        self.max_offset = float(max_offset)
        self.max_correction = float(max_correction)
        self.encoder = _conv_block(4, channels)
        self.offset_mask = nn.Conv2d(channels, 27, 3, padding=1)
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        self.deform = BoundedModulatedDeformConv2d(channels)
        self.residual = nn.Conv2d(channels, 1, 1)
        nn.init.normal_(self.residual.weight, std=1e-3)
        nn.init.zeros_(self.residual.bias)

    def forward(self, depth: torch.Tensor, reliability: torch.Tensor,
                size: tuple[int, int]) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type=depth.device.type, enabled=False):
            reliability = reliability.detach().float().clamp(0, 1)
            quality = F.adaptive_avg_pool2d(reliability, size)
            # Remove untrusted values before pooling, derivatives, or deformable sampling.
            masked = torch.where(reliability > 0, depth.float(), 0.0) * reliability
            base = F.adaptive_avg_pool2d(masked, size) / torch.where(
                quality > 0, quality, torch.ones_like(quality))
            dx, dy = _supported_derivatives(base, quality)
            feature = self.encoder(torch.cat((base, quality, dx, dy), dim=1))
            offset_mask = self.offset_mask(feature)
            offsets = self.max_offset * offset_mask[:, :18].tanh()
            modulation = offset_mask[:, 18:].sigmoid()
            deformed = self.deform(feature, offsets, modulation)
            correction = self.max_correction * self.residual(deformed).tanh()
            corrected = (base + correction).clamp(0, 1)
            trusted = corrected * quality
        return dict(corrected_depth=corrected, trusted_depth=trusted, base_depth=base,
                    reliability=quality, offsets=offsets, correction=correction)


class SmoothFrequencySplitV6(nn.Module):
    """Confidence-normalized Gaussian low-pass and complementary trusted detail.

    ``sigma = 1 / (2*pi*cutoff)`` in feature pixels. A positive spatial kernel
    is essential: the former super-Gaussian FFT kernel can have negative
    sidelobes and is not suitable for dividing by filtered confidence.
    """

    def __init__(self, cutoff: float = 0.15, padding: int = 8) -> None:
        super().__init__()
        if not 0.0 < cutoff <= 0.5:
            raise ValueError('frequency cutoff must be in (0, 0.5] cycles/pixel')
        if not isinstance(padding, int) or padding < 0:
            raise ValueError('frequency padding must be a nonnegative integer')
        self.cutoff = float(cutoff)
        self.padding = padding
        coordinate = torch.arange(-padding, padding + 1, dtype=torch.float64)
        sigma = 1.0 / (2.0 * math.pi * cutoff)
        kernel = torch.exp(-0.5 * (coordinate / sigma).square())
        self.register_buffer('kernel', (kernel / kernel.sum()).float(), persistent=False)

    def _filter(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        pad_h, pad_w = min(self.padding, height - 1), min(self.padding, width - 1)
        channels = value.shape[1]
        if pad_w:
            kernel = self.kernel[self.padding - pad_w:self.padding + pad_w + 1]
            kernel = (kernel / kernel.sum()).view(1, 1, 1, -1).expand(channels, 1, 1, -1)
            value = F.conv2d(F.pad(value, (pad_w, pad_w, 0, 0), mode='reflect'),
                             kernel, groups=channels)
        if pad_h:
            kernel = self.kernel[self.padding - pad_h:self.padding + pad_h + 1]
            kernel = (kernel / kernel.sum()).view(1, 1, -1, 1).expand(channels, 1, -1, 1)
            value = F.conv2d(F.pad(value, (0, 0, pad_h, pad_h), mode='reflect'),
                             kernel, groups=channels)
        return value

    def forward(self, depth: torch.Tensor, reliability: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=depth.device.type, enabled=False):
            quality = (torch.ones_like(depth, dtype=torch.float32) if reliability is None
                       else reliability.detach().float().clamp(0, 1))
            value = torch.where(quality > 0, depth.float(), 0.0)
            trusted = value * quality
            filtered = self._filter(torch.cat((trusted, quality), dim=1))
            numerator, support = filtered.chunk(2, dim=1)
            mean = numerator / torch.where(support > 0, support, torch.ones_like(support))
            low = quality * mean
            high = trusted - low
        return low.to(depth.dtype), high.to(depth.dtype)


class SpatialGeometryClusteringV6(nn.Module):
    """Spatially regularized soft clustering with matched RGB region pooling."""

    def __init__(
        self,
        channels: int,
        grid_size: int = 4,
        max_cluster_size: int = 64,
        iterations: int = 2,
        temperature: float = 0.2,
        spatial_weight: float = 2.0,
        min_reliability: float = 0.35,
        min_region_mass: float = 2.0,
        min_region_energy: float = 1e-4,
    ) -> None:
        super().__init__()
        if min(channels, grid_size, max_cluster_size, iterations) < 1:
            raise ValueError('clustering dimensions and iterations must be positive')
        if (not math.isfinite(temperature) or temperature <= 0
                or not math.isfinite(spatial_weight) or spatial_weight < 0):
            raise ValueError('invalid clustering temperature or spatial weight')
        if not 0 <= min_reliability <= 1:
            raise ValueError('min_reliability must be in [0, 1]')
        if (not math.isfinite(min_region_mass) or min_region_mass < 0
                or not math.isfinite(min_region_energy) or min_region_energy < 0):
            raise ValueError('region mass and energy thresholds must be finite and nonnegative')
        self.grid_size = int(grid_size)
        self.max_cluster_size = int(max_cluster_size)
        self.iterations = int(iterations)
        self.temperature = float(temperature)
        self.spatial_weight = float(spatial_weight)
        self.min_reliability = float(min_reliability)
        self.min_region_mass = float(min_region_mass)
        self.min_region_energy = float(min_region_energy)
        self.context = _conv_block(channels * 2, channels)

    def forward(
        self,
        geometry: torch.Tensor,
        rgb: torch.Tensor,
        reliability: torch.Tensor,
        high_energy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        size = _bounded_size(geometry, self.max_cluster_size)
        grid = (min(self.grid_size, size[0]), min(self.grid_size, size[1]))
        # Region selection is a measurement, not a trainable way to evade the contrastive objective.
        with torch.autocast(device_type=geometry.device.type, enabled=False):
            geo = F.adaptive_avg_pool2d(geometry.float(), size)
            appearance = F.adaptive_avg_pool2d(rgb.float(), size).flatten(2)
            quality_map = F.adaptive_avg_pool2d(reliability.detach().float(), size)
            quality_map = quality_map.clamp(0, 1)
            quality = quality_map.flatten(2)
            energy = F.adaptive_avg_pool2d(high_energy.detach().float(), size).flatten(2)
            features = geo.flatten(2)
            descriptors = F.normalize(features, dim=1, eps=1e-6)
            coords = _coordinates(*size, geo).flatten(2).expand(geo.shape[0], -1, -1)
            seed_mass = F.adaptive_avg_pool2d(quality_map, grid).flatten(2)
            centers = F.adaptive_avg_pool2d(geo * quality_map, grid).flatten(2)
            centers = centers / seed_mass.clamp_min(1e-6)
            positions = _coordinates(*grid, geo).flatten(2).expand(geo.shape[0], -1, -1)

            for _ in range(self.iterations):
                similarity = torch.bmm(
                    F.normalize(centers, dim=1, eps=1e-6).transpose(1, 2), descriptors)
                distance = (positions.unsqueeze(-1) - coords.unsqueeze(2)).square().sum(1)
                assignments = ((similarity - self.spatial_weight * distance)
                               / self.temperature).softmax(dim=1)
                weights = assignments * quality
                mass = weights.sum(-1)
                pooling = weights / mass.unsqueeze(-1).clamp_min(1e-6)
                centers = torch.bmm(features, pooling.transpose(1, 2))
                positions = torch.bmm(coords, pooling.transpose(1, 2))

            # The contrastive objective aligns descriptors, rather than moving its own region assignments.
            contrastive_pooling = pooling.detach().transpose(1, 2)
            geometry_prototypes = torch.bmm(features, contrastive_pooling).transpose(1, 2)
            rgb_prototypes = torch.bmm(appearance, contrastive_pooling).transpose(1, 2)
            region_quality = mass / assignments.sum(-1).clamp_min(1e-6)
            region_energy = (pooling * energy).sum(-1)
            valid = ((mass > 0)
                     & (region_quality >= self.min_reliability)
                     & (mass >= self.min_region_mass)
                     & (region_energy >= self.min_region_energy)).detach()
            # Contrastive eligibility never switches off segmentation detail.
            dense = torch.bmm(centers, assignments).reshape(geo.shape)

        dense = _resize(dense.to(geometry.dtype), geometry.shape[-2:])
        detail = geometry + self.context(torch.cat((geometry, dense), dim=1))
        return dict(
            detail=detail,
            geometry_prototypes=geometry_prototypes,
            rgb_prototypes=rgb_prototypes,
            valid_regions=valid,
            region_reliability=region_quality.detach(),
            region_mass=mass.detach(),
            region_energy=region_energy.detach(),
        )


class RegionContrastiveLossV6(nn.Module):
    """SigLIP-style sigmoid alignment over reliable paired geometry/RGB regions."""

    def __init__(self, channels: int, embed_dim: int = 64, temperature: float = 0.1,
                 logit_bias: float = -10.0):
        super().__init__()
        if channels < 1 or embed_dim < 1 or not 0.01 <= temperature <= 1:
            raise ValueError('invalid contrastive dimensions or temperature')
        if not math.isfinite(logit_bias):
            raise ValueError('logit_bias must be finite')
        self.geometry_projection = nn.Sequential(
            nn.Linear(channels, channels), nn.LayerNorm(channels), nn.GELU(approximate='tanh'),
            nn.Linear(channels, embed_dim),
        )
        self.rgb_projection = nn.Sequential(
            nn.Linear(channels, channels), nn.LayerNorm(channels), nn.GELU(approximate='tanh'),
            nn.Linear(channels, embed_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.logit_bias = nn.Parameter(torch.tensor(float(logit_bias)))

    def forward(
        self,
        geometry: torch.Tensor,
        rgb: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=geometry.device.type, enabled=False):
            observed = valid.detach().bool().unsqueeze(-1)
            geometry_embeddings = F.normalize(
                self.geometry_projection(torch.where(observed, geometry.float(), 0.0)),
                dim=-1, eps=1e-6)
            rgb_embeddings = F.normalize(
                self.rgb_projection(torch.where(observed, rgb.float(), 0.0)), dim=-1, eps=1e-6)
            geometry_embeddings = geometry_embeddings.flatten(0, 1)
            rgb_embeddings = rgb_embeddings.flatten(0, 1)
            selected = valid.detach().flatten().bool()
            scale = self.logit_scale.clamp(0.0, math.log(100.0)).exp()
            logits = scale * geometry_embeddings @ rgb_embeddings.transpose(0, 1) + self.logit_bias
            positive = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
            signed_logits = torch.where(positive, logits, -logits)
            pair_valid = selected.unsqueeze(1) & selected.unsqueeze(0)
            pair_loss = -F.logsigmoid(signed_logits)
            count = selected.float().sum()
            return (pair_loss * pair_valid).sum() / count.clamp_min(1)


class LowFrequencyCrossAttentionV6(nn.Module):
    """RGB queries attend to bounded, reliability-weighted low-frequency tokens."""

    def __init__(self, channels: int, num_heads: int = 4,
                 query_size: int = 32, kv_size: int = 8) -> None:
        super().__init__()
        if channels < 1 or num_heads < 1 or channels % num_heads:
            raise ValueError('channels must be positive and divisible by num_heads')
        if min(query_size, kv_size) < 1:
            raise ValueError('attention token sizes must be positive')
        self.channels = channels
        self.num_heads = num_heads
        self.query_size = query_size
        self.kv_size = kv_size
        self.rgb_norm = nn.LayerNorm(channels)
        self.depth_norm = nn.LayerNorm(channels)
        self.position = nn.Linear(2, channels, bias=False)
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, rgb: torch.Tensor, low: torch.Tensor,
                reliability: torch.Tensor) -> torch.Tensor:
        batch = rgb.shape[0]
        query_size = _bounded_size(rgb, self.query_size)
        kv_size = _bounded_size(low, self.kv_size)
        rgb_tokens = F.adaptive_avg_pool2d(rgb, query_size).flatten(2).transpose(1, 2)
        # Pool values and confidence together so invalid pixels cannot dominate a key.
        quality = reliability.detach().to(low.dtype).clamp(0, 1)
        kv_quality = F.adaptive_avg_pool2d(quality, kv_size)
        low_tokens = F.adaptive_avg_pool2d(low * quality, kv_size)
        low_tokens = (low_tokens / kv_quality.clamp_min(1e-6)).flatten(2).transpose(1, 2)
        q_pos = self.position(_coordinates(*query_size, rgb).flatten(2).transpose(1, 2))
        k_pos = self.position(_coordinates(*kv_size, low).flatten(2).transpose(1, 2))
        q = self.query(self.rgb_norm(rgb_tokens) + q_pos)
        k = self.key(self.depth_norm(low_tokens) + k_pos)
        v = self.value(low_tokens)
        head_dim = self.channels // self.num_heads
        q = q.reshape(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        k = k.reshape(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(head_dim)
            confidence = kv_quality.float().flatten(2).unsqueeze(2)
            attention = (scores + confidence.clamp_min(1e-6).log()).softmax(dim=-1)
            attention = attention * (confidence > 0)
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-6)
            context = torch.matmul(attention, v.float())
        context = context.transpose(1, 2).reshape(batch, *query_size, self.channels)
        # Mixed NCHW/channels-last gradients corrupt CPU GELU backward in PyTorch 2.1.
        context = context.permute(0, 3, 1, 2).contiguous().to(rgb.dtype)
        return _resize(self.output(context), rgb.shape[-2:])


class GatedGeometryFusionV6(nn.Module):
    """Confidence-gated geometry residuals preserve an exact RGB fallback."""

    def __init__(self, channels: int, residual_scale: float = 0.1) -> None:
        super().__init__()
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError('residual_scale must be finite and nonnegative')
        self.branch_gains = nn.Parameter(torch.full((2,), float(residual_scale)))
        self.gates = nn.Sequential(
            nn.Conv2d(channels * 3 + 1, channels, 1), nn.GELU(approximate='tanh'),
            nn.Conv2d(channels, 2, 1),
        )
        nn.init.normal_(self.gates[-1].weight, std=1e-3)
        nn.init.zeros_(self.gates[-1].bias)

    def forward(self, rgb: torch.Tensor, high: torch.Tensor, low: torch.Tensor,
                reliability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reliability = reliability.detach().float().clamp(0, 1)
        gates = self.gates(torch.cat((rgb, high, low, reliability.to(rgb.dtype)), dim=1)).sigmoid()
        update = (self.branch_gains[0] * gates[:, :1].float() * high.float()
                  + self.branch_gains[1] * gates[:, 1:].float() * low.float())
        return reliability * update, gates


class FrequencyGeometryStageV6(nn.Module):
    """A geometry stage with independent appearance references and delta output."""

    def __init__(self, channels: int, cluster_cfg: dict | None = None,
                 attention_cfg: dict | None = None, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.high_encoder = _conv_block(1, channels)
        self.low_encoder = _conv_block(1, channels)
        self.clustering = SpatialGeometryClusteringV6(channels, **(cluster_cfg or {}))
        self.low_attention = LowFrequencyCrossAttentionV6(channels, **(attention_cfg or {}))
        self.fusion = GatedGeometryFusionV6(channels, residual_scale)

    def forward(self, rgb: torch.Tensor, low_depth: torch.Tensor,
                high_depth: torch.Tensor, reliability: torch.Tensor,
                rgb_reference: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        size = rgb.shape[-2:]
        low = _resize(low_depth, size)
        high = _resize(high_depth, size)
        reliability = _resize(reliability, size).detach().clamp(0, 1)
        appearance = rgb if rgb_reference is None else rgb_reference
        regions = self.clustering(self.high_encoder(high), appearance, reliability,
                                  _resize(high_depth.abs(), size))
        low_context = self.low_attention(rgb, self.low_encoder(low), reliability)
        delta, gates = self.fusion(rgb, regions['detail'], low_context, reliability)
        return dict(regions, delta=delta, gates=gates)
