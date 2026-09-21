"""Reusable building blocks for the RPGV-Net geometry branch."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def binary_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalized binary entropy, evaluated in FP32 even under autocast.

    H(sigmoid(z)) = softplus(-|z|) + |z| * sigmoid(-|z|).
    Unlike probability-space log/clamp, this stays finite for saturated
    finite logits and never relies on representing 1 - epsilon in FP16.
    """
    magnitude = logits.float().abs()
    return (
        F.softplus(-magnitude) + magnitude * torch.sigmoid(-magnitude)
    ) / math.log(2.0)


def _group_norm(channels: int) -> nn.GroupNorm:
    """Return a batch-size independent normalization layer."""
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, groups=groups, bias=False),
            _group_norm(out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


class DepthwiseSeparableBlock(nn.Module):
    """Lightweight spatial block used by the geometry encoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(
                in_channels, in_channels, kernel_size=3, stride=stride,
                groups=in_channels),
            ConvNormAct(in_channels, out_channels, kernel_size=1),
        )
        self.shortcut = (
            ConvNormAct(
                in_channels, out_channels, kernel_size=1, stride=stride,
                activation=False)
            if stride != 1 or in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.block(x) + self.shortcut(x))


class GlobalContextFiLM(nn.Module):
    """Identity-initialized FiLM modulation from a pooled scene token."""

    def __init__(
        self,
        token_channels: int,
        feature_channels: Sequence[int],
    ) -> None:
        super().__init__()
        self.modulators = nn.ModuleList([
            nn.Linear(token_channels, 2 * channels)
            for channels in feature_channels
        ])
        for modulator in self.modulators:
            nn.init.zeros_(modulator.weight)
            nn.init.zeros_(modulator.bias)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        token: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        modulated = []
        for feature, modulator in zip(features, self.modulators):
            gamma, beta = modulator(token).chunk(2, dim=1)
            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]
            modulated.append((1.0 + gamma) * feature + beta)
        return tuple(modulated)


def spatial_derivatives(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Sobel first derivatives, magnitude and four-neighbour Laplacian."""
    dtype, device = x.dtype, x.device
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype, device=device).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    laplace = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=dtype, device=device).view(1, 1, 3, 3)
    padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
    dx = F.conv2d(padded, sobel_x)
    dy = F.conv2d(padded, sobel_y)
    magnitude = torch.sqrt(dx.square() + dy.square() + 1e-6)
    curvature = F.conv2d(padded, laplace)
    return dx, dy, magnitude, curvature


class BoundedModulatedDeformConv2d(nn.Module):
    """Portable DCNv2-style 3x3 sampling implemented with ``grid_sample``.

    MMCV's compiled modulated deformable convolution is not available on all
    CPU/deployment targets used by this project.  This implementation keeps
    the important behaviour (learned per-kernel offsets and modulation) while
    remaining differentiable and backend independent.  Offsets are supplied
    in feature pixels and are expected to be bounded by the caller.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size != 3:
            raise ValueError('only a 3x3 deformable kernel is supported')
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(channels, channels, kernel_size * kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @staticmethod
    def _base_grid(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        modulation: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = x.shape
        if offsets.shape[1] != 18 or modulation.shape[1] != 9:
            raise ValueError('3x3 deformable convolution needs 18 offsets and 9 masks')
        offsets = offsets.view(batch, 9, 2, height, width)
        base_grid = self._base_grid(height, width, x.device, x.dtype)
        scale_x = 2.0 / max(width - 1, 1)
        scale_y = 2.0 / max(height - 1, 1)
        output = x.new_zeros(batch, self.channels, height, width)

        for index, (kernel_y, kernel_x) in enumerate(
            (y, x_) for y in (-1, 0, 1) for x_ in (-1, 0, 1)
        ):
            offset_y = offsets[:, index, 0]
            offset_x = offsets[:, index, 1]
            grid = base_grid.unsqueeze(0).expand(batch, -1, -1, -1).clone()
            grid[..., 0] += (kernel_x + offset_x) * scale_x
            grid[..., 1] += (kernel_y + offset_y) * scale_y
            sampled = F.grid_sample(
                x, grid, mode='bilinear', padding_mode='border',
                align_corners=True)
            sampled = sampled * modulation[:, index:index + 1]
            output += F.conv2d(
                sampled, self.weight[:, :, index, None, None], bias=None)
        return output + self.bias.view(1, -1, 1, 1)


class ReliabilityGuidedRectifier(nn.Module):
    """Reliability-aware bounded correction of the pseudo depth map."""

    def __init__(
        self,
        rgb_channels: int,
        channels: int = 32,
        max_offset: float = 2.0,
        correction_scale: float = 0.1,
        max_correction_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if not 0.0 < correction_scale < max_correction_scale:
            raise ValueError('correction_scale must lie inside its configured bound')
        self.max_offset = max_offset
        self.max_correction_scale = max_correction_scale
        initial_logit = math.log(
            correction_scale / (max_correction_scale - correction_scale))
        self.correction_logit = nn.Parameter(torch.tensor(initial_logit))
        self.depth_encoder = nn.Sequential(
            ConvNormAct(4, channels),
            ConvNormAct(channels, channels),
        )
        self.rgb_projection = ConvNormAct(
            rgb_channels, channels, kernel_size=1)
        self.offset_mask = nn.Sequential(
            ConvNormAct(2 * channels + 1, channels),
            nn.Conv2d(channels, 27, 3, padding=1),
        )
        self.deform = BoundedModulatedDeformConv2d(channels)
        self.learned_reliability = nn.Sequential(
            ConvNormAct(2 * channels + 1, channels),
            nn.Conv2d(channels, 1, 1),
        )
        self.residual = nn.Sequential(
            # RGB contributes only its explicit boundary cue here.  This lets
            # the rectifier repair locally flat/broken depth without exposing
            # the full semantic RGB feature as an unconstrained shortcut.
            ConvNormAct(channels + 1, channels),
            nn.Conv2d(channels, 1, 1),
        )

        # An initially centred sampler makes early optimization stable.
        nn.init.zeros_(self.offset_mask[-1].weight)
        nn.init.zeros_(self.offset_mask[-1].bias)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        # Q_learn is an attenuation factor on the offline prior Q0.  Starting
        # near one preserves Q0 instead of accidentally squaring/suppressing it
        # before the reliability head has learned anything.
        nn.init.zeros_(self.learned_reliability[-1].weight)
        nn.init.constant_(
            self.learned_reliability[-1].bias, math.log(0.95 / 0.05))

    @property
    def correction_scale(self) -> torch.Tensor:
        return self.max_correction_scale * self.correction_logit.sigmoid()

    def forward(
        self,
        depth: torch.Tensor,
        reliability: torch.Tensor,
        rgb_feature: torch.Tensor,
        rgb_boundary: torch.Tensor,
        use_depth_correction: bool = True,
        use_learned_reliability: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Rectify pseudo depth and calibrate its reliability.

        The two switches deliberately live inside the module rather than in
        separate ablation-only implementations.  This keeps checkpoint keys
        identical between the full model and every ablation while allowing
        correction and reliability calibration to be studied independently.
        """
        size = rgb_feature.shape[-2:]
        depth = F.interpolate(
            depth, size=size, mode='bilinear', align_corners=False)
        reliability = F.interpolate(
            reliability, size=size, mode='bilinear', align_corners=False)
        if not use_depth_correction and not use_learned_reliability:
            # Complete RGR ablation: preserve raw pseudo geometry without
            # spending compute in the otherwise fully frozen rectifier.
            learned = torch.ones_like(reliability, dtype=torch.float32)
            return dict(
                corrected_depth=depth,
                base_depth=depth,
                reliability=reliability,
                learned_reliability=learned,
                learned_reliability_logits=torch.full_like(learned, 20.0),
                offsets=depth.new_zeros(depth.shape[0], 18, *size),
            )
        dx, dy, magnitude, _ = spatial_derivatives(depth)
        depth_feature = self.depth_encoder(
            torch.cat([depth, dx, dy, magnitude], dim=1))
        rgb_feature = self.rgb_projection(rgb_feature)
        joint = torch.cat([depth_feature, rgb_feature, rgb_boundary], dim=1)
        offset_mask = self.offset_mask(joint)
        offsets = self.max_offset * torch.tanh(offset_mask[:, :18])
        modulation = offset_mask[:, 18:].sigmoid()
        deformed = self.deform(depth_feature, offsets, modulation)
        if use_learned_reliability:
            learned_logits = self.learned_reliability(
                torch.cat([deformed, rgb_feature, reliability], dim=1))
            learned = learned_logits.float().sigmoid()
            task_reliability = reliability * learned
        else:
            # A finite constant keeps diagnostic outputs well behaved.  The
            # corresponding supervision is omitted by RPGVNet in this mode.
            learned = torch.ones_like(reliability, dtype=torch.float32)
            learned_logits = torch.full_like(learned, 20.0)
            task_reliability = reliability

        if use_depth_correction:
            correction = torch.tanh(self.residual(torch.cat([
                deformed, rgb_boundary,
            ], dim=1)))
            corrected = torch.clamp(
                depth + self.correction_scale
                * task_reliability.detach() * correction,
                0.0, 1.0)
        else:
            corrected = depth
        return dict(
            corrected_depth=corrected,
            base_depth=depth,
            reliability=task_reliability,
            learned_reliability=learned,
            learned_reliability_logits=learned_logits,
            offsets=offsets,
        )


class GeometryEncoder(nn.Module):
    """Encode depth, differential geometry and local relief at four scales."""

    def __init__(self, channels: Sequence[int] = (32, 64, 128, 256)):
        super().__init__()
        if len(channels) != 4:
            raise ValueError('GeometryEncoder requires four channel stages')
        self.channels = tuple(channels)
        self.stem = nn.Sequential(
            ConvNormAct(8, channels[0]),
            DepthwiseSeparableBlock(channels[0], channels[0]),
        )
        self.downsamples = nn.ModuleList([
            DepthwiseSeparableBlock(channels[i], channels[i + 1], stride=2)
            for i in range(3)
        ])

    def forward(
        self, depth: torch.Tensor, reliability: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        dx, dy, magnitude, curvature = spatial_derivatives(depth)
        relief3 = depth - F.avg_pool2d(depth, 3, stride=1, padding=1)
        relief7 = depth - F.avg_pool2d(depth, 7, stride=1, padding=3)
        descriptor = torch.cat([
            depth, dx, dy, magnitude, curvature, relief3, relief7, reliability
        ], dim=1)
        features = [self.stem(descriptor)]
        for downsample in self.downsamples:
            features.append(downsample(features[-1]))
        return tuple(features)


class GeometryFeaturePyramid(nn.Module):
    """Propagate deep geometry context to every prediction scale.

    The geometry pretraining head operates at stride four.  Without a
    top-down path that loss cannot reach the deeper encoder stages.  This
    lightweight pyramid keeps the original channel contract while making all
    four stages useful to both the auxiliary geometry task and joint fusion.
    """

    def __init__(self, channels: Sequence[int] = (32, 64, 128, 256)):
        super().__init__()
        if len(channels) != 4:
            raise ValueError('GeometryFeaturePyramid requires four stages')
        self.laterals = nn.ModuleList([
            ConvNormAct(value, value, kernel_size=1) for value in channels
        ])
        self.top_down = nn.ModuleList([
            ConvNormAct(channels[index + 1], channels[index], kernel_size=1)
            for index in range(3)
        ])
        self.refine = nn.ModuleList([
            DepthwiseSeparableBlock(channels[index], channels[index])
            for index in range(3)
        ])

    def forward(
        self, features: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        if len(features) != 4:
            raise ValueError('GeometryFeaturePyramid expects four features')
        outputs = [
            lateral(feature)
            for lateral, feature in zip(self.laterals, features)
        ]
        for index in range(2, -1, -1):
            context = self.top_down[index](outputs[index + 1])
            context = F.interpolate(
                context, size=outputs[index].shape[-2:], mode='bilinear',
                align_corners=False)
            outputs[index] = self.refine[index](outputs[index] + context)
        return tuple(outputs)


class HaarWavelet2D(nn.Module):
    """Parameter-free one-level orthonormal 2-D Haar transform."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = x.shape[-2:]
        if height % 2 or width % 2:
            x = F.pad(x, (0, width % 2, 0, height % 2), mode='replicate')
        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]
        low = (a + b + c + d) * 0.5
        lh = (-a - b + c + d) * 0.5
        hl = (-a + b - c + d) * 0.5
        hh = (a - b - c + d) * 0.5
        return low, torch.cat([lh, hl, hh], dim=1)

    def inverse(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        lh, hl, hh = high.chunk(3, dim=1)
        a = (low - lh - hl + hh) * 0.5
        b = (low - lh + hl - hh) * 0.5
        c = (low + lh - hl - hh) * 0.5
        d = (low + lh + hl + hh) * 0.5
        batch, channels, height, width = low.shape
        result = low.new_empty(batch, channels, 2 * height, 2 * width)
        result[..., 0::2, 0::2] = a
        result[..., 0::2, 1::2] = b
        result[..., 1::2, 0::2] = c
        result[..., 1::2, 1::2] = d
        if output_size is not None:
            result = result[..., :output_size[0], :output_size[1]]
        return result


class DualFrequencyGeometryValidator(nn.Module):
    """Validate geometry separately in the boundary and region bands."""

    def __init__(
        self,
        rgb_channels: int,
        geometry_channels: int,
        region_rgb_channels: int,
        region_geometry_channels: int,
        channels: int = 32,
    ) -> None:
        super().__init__()
        self.channels = channels
        # Keep the original attribute names so older checkpoints can restore
        # the already trained high-frequency path.
        self.rgb_projection = ConvNormAct(
            rgb_channels, channels, kernel_size=1)
        self.geometry_projection = ConvNormAct(
            geometry_channels, channels, kernel_size=1)
        self.region_rgb_projection = ConvNormAct(
            region_rgb_channels, channels, kernel_size=1)
        self.region_geometry_projection = ConvNormAct(
            region_geometry_channels, channels, kernel_size=1)
        self.wavelet = HaarWavelet2D()
        self.high_validator = nn.Sequential(
            ConvNormAct(9 * channels + 2, 3 * channels),
            nn.Conv2d(3 * channels, 3 * channels, 1),
        )
        self.high_projection = nn.Conv2d(3 * channels, 3 * channels, 1)
        self.region_validator = nn.Sequential(
            ConvNormAct(3 * channels + 2, channels),
            nn.Conv2d(channels, channels, 1),
        )
        self.low_projection = nn.Conv2d(channels, channels, 1)

    @staticmethod
    def _resize(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(value, size=size, mode='bilinear', align_corners=False)

    def boundary_delta(
        self,
        rgb: torch.Tensor,
        geometry: torch.Tensor,
        reliability: torch.Tensor,
        rgb_boundary: torch.Tensor,
    ) -> torch.Tensor:
        """Return the validated stride-4 Haar high-frequency residual."""
        rgb = self.rgb_projection(rgb)
        geometry = self.geometry_projection(geometry)
        low_reference, high_rgb = self.wavelet(rgb)
        _, high_geo = self.wavelet(geometry)
        band_size = low_reference.shape[-2:]
        high_reliability = self._resize(reliability, band_size)
        boundary = self._resize(rgb_boundary, band_size)

        high_weight = self.high_validator(torch.cat([
            high_rgb, high_geo, (high_rgb - high_geo).abs(),
            high_reliability, boundary,
        ], dim=1)).sigmoid()
        # Reliability is applied once, at the final residual fusion.  It still
        # informs frequency validation, but does not quadratically attenuate
        # the candidate geometry residual.
        high_delta = high_weight * self.high_projection(high_geo)
        return self.wavelet.inverse(
            torch.zeros_like(low_reference), high_delta, rgb.shape[-2:])

    def region_delta(
        self,
        region_rgb: torch.Tensor,
        region_geometry: torch.Tensor,
        reliability: torch.Tensor,
        rgb_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        """Return the validated stride-16 semantic region residual."""
        # Region completion uses the semantic stride-16 features directly.
        # The previous implementation reused the stride-8 LL band from G1,
        # leaving deep geometry without a content path into the decoder.
        region_rgb = self.region_rgb_projection(region_rgb)
        region_geometry = self.region_geometry_projection(region_geometry)
        region_size = region_rgb.shape[-2:]
        region_reliability = self._resize(reliability, region_size)
        region_uncertainty = self._resize(rgb_uncertainty, region_size)
        low_weight = self.region_validator(torch.cat([
            region_rgb, region_geometry,
            (region_rgb - region_geometry).abs(),
            region_reliability, region_uncertainty,
        ], dim=1)).sigmoid()
        return low_weight * self.low_projection(region_geometry)

    def direct_boundary_delta(self, geometry: torch.Tensor) -> torch.Tensor:
        """Project stride-4 geometry without frequency decomposition."""
        return self.geometry_projection(geometry)

    def direct_region_delta(
        self, region_geometry: torch.Tensor
    ) -> torch.Tensor:
        """Project stride-16 geometry without cross-modal validation."""
        return self.region_geometry_projection(region_geometry)

    def forward(
        self,
        rgb: torch.Tensor,
        geometry: torch.Tensor,
        region_rgb: torch.Tensor,
        region_geometry: torch.Tensor,
        reliability: torch.Tensor,
        rgb_boundary: torch.Tensor,
        rgb_uncertainty: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate both residual bands for the default full model."""
        return (
            self.boundary_delta(
                rgb, geometry, reliability, rgb_boundary),
            self.region_delta(
                region_rgb, region_geometry, reliability, rgb_uncertainty),
        )


class ReliabilityWeightedResidualFusion(nn.Module):
    """Inject a validated geometry residual in proportion to reliability."""

    def __init__(
        self,
        rgb_channels: int,
        delta_channels: int,
    ) -> None:
        super().__init__()
        self.delta_projection = ConvNormAct(
            delta_channels, rgb_channels, kernel_size=1, activation=False)
        # Stage two freezes this module. A zero projection guarantees exact
        # RGB behaviour on the first joint-training step while allowing the
        # final segmentation loss to learn a useful residual immediately.
        nn.init.zeros_(self.delta_projection[0].weight)

    @staticmethod
    def _resize(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(value, size=size, mode='bilinear', align_corners=False)

    def forward(
        self,
        rgb: torch.Tensor,
        delta: torch.Tensor,
        reliability: torch.Tensor,
        use_reliability: bool = True,
    ) -> torch.Tensor:
        size = rgb.shape[-2:]
        delta = self._resize(delta, size)
        residual = torch.tanh(self.delta_projection(delta))
        if use_reliability:
            reliability = self._resize(reliability, size)
            residual = reliability * residual
        return rgb + residual


class MultiScaleDecoder(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int = 128,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList([
            ConvNormAct(value, channels, kernel_size=1)
            for value in in_channels
        ])
        self.fuse = nn.Sequential(
            ConvNormAct(len(in_channels) * channels, channels),
            ConvNormAct(channels, channels),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        output_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != output_size:
                feature = F.interpolate(
                    feature, size=output_size, mode='bilinear',
                    align_corners=False)
            projected.append(feature)
        return self.fuse(torch.cat(projected, dim=1))


class BoundaryResidualRefiner(nn.Module):
    """Predict coarse/boundary/SDF heads and refine only where warranted."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.coarse_head = nn.Conv2d(channels, 1, 1)
        self.boundary_head = nn.Sequential(
            ConvNormAct(channels, channels // 2),
            nn.Conv2d(channels // 2, 1, 1),
        )
        self.sdf_head = nn.Sequential(
            ConvNormAct(channels, channels // 2),
            nn.Conv2d(channels // 2, 1, 1),
        )
        self.boundary_gate = nn.Sequential(
            ConvNormAct(4, 16),
            nn.Conv2d(16, 1, 1),
        )
        self.refinement = nn.Sequential(
            ConvNormAct(channels + 3, channels),
            nn.Conv2d(channels, 1, 1),
        )
        # Stage one always supplies zero reliability.  Zero the corresponding
        # input slice so a non-zero Qd in stage three cannot activate a random,
        # never-trained coefficient and disturb the RGB checkpoint.
        nn.init.zeros_(self.boundary_gate[0][0].weight[:, 3:4])
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)

    def forward(
        self, feature: torch.Tensor, reliability: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        coarse = self.coarse_head(feature)
        boundary = self.boundary_head(feature)
        sdf = self.sdf_head(feature)
        uncertainty = binary_entropy_from_logits(coarse)
        reliability = F.interpolate(
            reliability, size=feature.shape[-2:], mode='bilinear',
            align_corners=False)
        gate = self.boundary_gate(torch.cat([
            boundary.sigmoid(), 1.0 - sdf.tanh().abs(), uncertainty,
            reliability,
        ], dim=1)).sigmoid()
        residual = self.refinement(torch.cat([
            feature, coarse, boundary, sdf,
        ], dim=1))
        return dict(
            final=coarse + gate * residual,
            coarse=coarse,
            boundary=boundary,
            sdf=sdf,
            refinement_gate=gate,
        )


class HighResolutionDetailRefiner(nn.Module):
    """Recover half-resolution RGB detail after stride-four decoding."""

    def __init__(self, decoder_channels: int, channels: int = 32) -> None:
        super().__init__()
        self.rgb_stem = nn.Sequential(
            ConvNormAct(3, channels, stride=2),
            DepthwiseSeparableBlock(channels, channels),
        )
        self.decoder_projection = ConvNormAct(
            decoder_channels, channels, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvNormAct(2 * channels + 1, channels),
            DepthwiseSeparableBlock(channels, channels),
        )
        self.boundary_head = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, 1, 1),
        )
        self.gate = nn.Sequential(
            ConvNormAct(3, 16),
            nn.Conv2d(16, 1, 1),
        )
        self.residual = nn.Sequential(
            ConvNormAct(channels + 2, channels),
            nn.Conv2d(channels, 1, 1),
        )
        # Preserve the trained stride-four prediction when this module is
        # introduced or loaded on top of an older checkpoint.
        # As in BoundaryResidualRefiner, Q is always zero during RGB training.
        nn.init.zeros_(self.gate[0][0].weight[:, 2:3])
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        rgb: torch.Tensor,
        decoder_feature: torch.Tensor,
        base_logits: torch.Tensor,
        reliability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rgb_detail = self.rgb_stem(rgb)
        size = rgb_detail.shape[-2:]
        decoder_detail = F.interpolate(
            self.decoder_projection(decoder_feature), size=size,
            mode='bilinear', align_corners=False)
        base_logits = F.interpolate(
            base_logits, size=size, mode='bilinear', align_corners=False)
        detail = self.fuse(torch.cat([
            rgb_detail, decoder_detail, base_logits.sigmoid(),
        ], dim=1))
        boundary = self.boundary_head(detail)
        uncertainty = binary_entropy_from_logits(base_logits)
        reliability = F.interpolate(
            reliability, size=size, mode='bilinear', align_corners=False)
        gate = self.gate(torch.cat([
            boundary.sigmoid(), uncertainty, reliability,
        ], dim=1)).sigmoid()
        residual = self.residual(torch.cat([
            detail, base_logits, boundary,
        ], dim=1))
        return dict(
            final=base_logits + gate * residual,
            boundary=boundary,
            refinement_gate=gate,
        )
