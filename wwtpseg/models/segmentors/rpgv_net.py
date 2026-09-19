"""RPGV-Net: reliability-aware pseudo-geometry segmentation model."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import numpy as np
import torch
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from torch import nn
from torch.nn import functional as F

from ..utils import (
    BoundaryResidualRefiner,
    DualFrequencyGeometryValidator,
    GeometryEncoder,
    GlobalContextFiLM,
    MultiScaleDecoder,
    ReliabilityGuidedRectifier,
    UncertaintyGatedResidualFusion,
)


def _binary_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Masked BCE + soft Dice for a one-logit binary prediction."""
    target = target.float().clamp(0.0, 1.0)
    valid = valid.float()
    denominator = valid.sum().clamp_min(1.0)
    bce = (
        F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        * valid).sum() / denominator
    probability = logits.sigmoid() * valid
    target = target * valid
    intersection = (probability * target).sum(dim=(1, 2, 3))
    cardinality = (probability + target).sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()
    return bce_weight * bce + dice_weight * dice


def _boundary_target(mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """One-pixel morphological boundary target at the prediction scale."""
    foreground = mask.float().clamp(0.0, 1.0)
    dilated = F.max_pool2d(foreground, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-foreground, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0) * valid.float()


def _signed_distance_target(
    mask: torch.Tensor,
    valid: torch.Tensor,
    truncation: float = 16.0,
) -> torch.Tensor:
    """Compute a truncated signed distance target on the CPU.

    Auxiliary heads operate at stride 4, so exact EDT is inexpensive and
    provides a substantially cleaner target than repeated pooling.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as error:  # pragma: no cover - runtime dependency guard
        raise RuntimeError('SciPy is required for the RPGV SDF loss') from error

    masks = (mask.detach().cpu().numpy() > 0.5)
    valids = (valid.detach().cpu().numpy() > 0.5)
    targets = []
    for sample, sample_valid in zip(masks, valids):
        foreground = sample[0] & sample_valid[0]
        background = (~foreground) & sample_valid[0]
        inside = distance_transform_edt(foreground)
        outside = distance_transform_edt(background)
        signed = np.clip((inside - outside) / truncation, -1.0, 1.0)
        signed[~sample_valid[0]] = 0.0
        targets.append(torch.from_numpy(signed.astype(np.float32)))
    return torch.stack(targets, dim=0).unsqueeze(1).to(mask.device)


@MODELS.register_module()
class RPGVNet(BaseSegmentor):
    """Reliability-aware Pseudo-Geometry Rectification and Validation Net.

    Geometry stages receive BGR plus offline normalized pseudo depth and
    reliability; the RGB pretraining stage receives BGR only.  The geometry
    branch is strictly residual, so a clean RGB path remains available when
    pseudo geometry is uninformative.
    """

    def __init__(
        self,
        rgb_encoder: dict,
        rgb_channels: Sequence[int] = (64, 128, 320, 512),
        geometry_channels: Sequence[int] = (32, 64, 128, 256),
        validation_channels: int = 32,
        decoder_channels: int = 128,
        correction_scale: float = 0.1,
        max_correction_scale: float = 0.25,
        max_offset: float = 2.0,
        gate_temperature: float = 0.5,
        gate_margin: float = 0.05,
        sdf_truncation: float = 5.0,
        geometry_dropout_prob: float = 0.15,
        training_stage: str = 'joint',
        use_global_context: bool = True,
        global_thumbnail_size: int = 512,
        loss_weights: dict | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if len(rgb_channels) != 4 or len(geometry_channels) != 4:
            raise ValueError('RPGVNet requires four RGB and geometry stages')
        if training_stage not in {'rgb', 'geometry', 'joint'}:
            raise ValueError(
                'training_stage must be one of rgb, geometry or joint')
        if global_thumbnail_size <= 0:
            raise ValueError('global_thumbnail_size must be positive')
        if gate_temperature <= 0.0 or sdf_truncation <= 0.0:
            raise ValueError(
                'gate_temperature and sdf_truncation must be positive')
        if gate_margin < 0.0:
            raise ValueError('gate_margin must be non-negative')
        if not 0.0 <= geometry_dropout_prob <= 1.0:
            raise ValueError('geometry_dropout_prob must lie in [0, 1]')
        self.rgb_channels = tuple(rgb_channels)
        self.geometry_channels = tuple(geometry_channels)
        self.training_stage = training_stage
        self.use_global_context = use_global_context
        self.global_thumbnail_size = global_thumbnail_size
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or dict(mode='whole')
        self.align_corners = False
        self.gate_temperature = gate_temperature
        self.gate_margin = gate_margin
        self.sdf_truncation = sdf_truncation
        self.geometry_dropout_prob = geometry_dropout_prob
        self.loss_weights = dict(
            final=1.0,
            rgb=0.3,
            geometry=0.2,
            boundary=0.2,
            sdf=0.1,
            reliability=0.05,
            preserve=0.05,
            equivariance=0.05,
            gate=0.1,
        )
        if loss_weights:
            unknown = set(loss_weights) - set(self.loss_weights)
            if unknown:
                raise ValueError(f'unknown RPGV loss weights: {sorted(unknown)}')
            self.loss_weights.update(loss_weights)

        self.rgb_encoder = MODELS.build(rgb_encoder)
        self.global_film = GlobalContextFiLM(
            rgb_channels[-1], rgb_channels)
        self.rgb_aux_head = nn.Conv2d(rgb_channels[0], 1, 1)
        self.rgb_boundary_head = nn.Sequential(
            nn.Conv2d(rgb_channels[0], validation_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(validation_channels, 1, 1),
        )
        self.rectifier = ReliabilityGuidedRectifier(
            rgb_channels=rgb_channels[0],
            channels=geometry_channels[0],
            max_offset=max_offset,
            correction_scale=correction_scale,
            max_correction_scale=max_correction_scale,
        )
        self.geometry_encoder = GeometryEncoder(geometry_channels)
        self.geometry_aux_head = nn.Conv2d(geometry_channels[0], 1, 1)
        self.frequency_validator = DualFrequencyGeometryValidator(
            rgb_channels[0], geometry_channels[0], validation_channels)
        self.high_fusion = UncertaintyGatedResidualFusion(
            rgb_channels[0], geometry_channels[0], validation_channels,
            validation_channels)
        self.low_fusion = UncertaintyGatedResidualFusion(
            rgb_channels[2], geometry_channels[2], validation_channels,
            validation_channels)
        self.decoder = MultiScaleDecoder(rgb_channels, decoder_channels)
        self.refiner = BoundaryResidualRefiner(decoder_channels)

        # SegDataPreProcessor leaves input channels untouched. Normalize only
        # BGR here and preserve optional geometry in [0, 1].
        self.register_buffer(
            'rgb_mean',
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'rgb_std',
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
            persistent=False)
        self._configure_trainable_parameters()

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def _configure_trainable_parameters(self) -> None:
        """Expose only the modules optimized by the selected training stage."""
        for parameter in self.parameters():
            parameter.requires_grad = self.training_stage == 'joint'
        if self.training_stage == 'rgb':
            for module in (
                self.rgb_encoder,
                self.global_film,
                self.rgb_aux_head,
                self.rgb_boundary_head,
                self.decoder,
                self.refiner,
            ):
                self._set_trainable(module, True)
        elif self.training_stage == 'geometry':
            for module in (
                self.rectifier,
                self.geometry_encoder,
                self.geometry_aux_head,
            ):
                self._set_trainable(module, True)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.training_stage == 'geometry':
            # Stage two treats the RGB expert as a deterministic teacher.
            for module in (
                self.rgb_encoder,
                self.global_film,
                self.rgb_aux_head,
                self.rgb_boundary_head,
                self.decoder,
                self.refiner,
                self.frequency_validator,
                self.high_fusion,
                self.low_fusion,
            ):
                module.eval()
        return self

    def _split_inputs(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        minimum_channels = 3 if self.training_stage == 'rgb' else 5
        if inputs.ndim != 4 or inputs.shape[1] < minimum_channels:
            raise ValueError(
                f'RPGVNet stage {self.training_stage!r} expects at least '
                f'{minimum_channels} channels; received {tuple(inputs.shape)}')
        if inputs.shape[1] not in (3, 5):
            raise ValueError(
                'RPGVNet inputs must contain BGR or BGR + pseudo depth and '
                f'reliability; received {tuple(inputs.shape)}')
        inputs = inputs.float()
        # LoadImageFromFile yields BGR.  MiT ImageNet weights expect RGB.
        rgb = self._normalize_bgr(inputs[:, :3])
        if inputs.shape[1] == 5:
            depth = (inputs[:, 3:4] / 255.0).clamp(0.0, 1.0)
            reliability = (inputs[:, 4:5] / 255.0).clamp(0.0, 1.0)
        else:
            depth = inputs.new_zeros(inputs.shape[0], 1, *inputs.shape[-2:])
            reliability = torch.zeros_like(depth)
        return rgb, depth, reliability

    def _normalize_bgr(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float().flip(1)
        return (images - self.rgb_mean) / self.rgb_std

    def _global_images_from_samples(
        self,
        inputs: torch.Tensor,
        data_samples: Sequence | None,
    ) -> torch.Tensor:
        if data_samples and all(
            hasattr(sample, 'global_img') for sample in data_samples
        ):
            return torch.stack([
                sample.global_img.data for sample in data_samples
            ], dim=0).to(device=inputs.device, dtype=inputs.dtype)
        return F.interpolate(
            inputs[:, :3].float(),
            size=(self.global_thumbnail_size, self.global_thumbnail_size),
            mode='bilinear',
            align_corners=False)

    def _global_token(self, global_images: torch.Tensor) -> torch.Tensor:
        global_features = self.rgb_encoder(self._normalize_bgr(global_images))
        return global_features[-1].mean(dim=(2, 3))

    def _encode_rgb(
        self,
        inputs: torch.Tensor,
        global_images: torch.Tensor | None = None,
        global_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        rgb, _, _ = self._split_inputs(inputs)
        features = tuple(self.rgb_encoder(rgb))
        if len(features) != 4:
            raise RuntimeError('rgb_encoder must return exactly four stages')
        if not self.use_global_context:
            return features
        if global_token is None:
            if global_images is None:
                global_images = F.interpolate(
                    inputs[:, :3].float(),
                    size=(
                        self.global_thumbnail_size,
                        self.global_thumbnail_size),
                    mode='bilinear',
                    align_corners=False)
            global_token = self._global_token(global_images)
        return self.global_film(features, global_token)

    def extract_feat(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self._encode_rgb(inputs)

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        probability = logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        return -(
            probability * probability.log()
            + (1.0 - probability) * (1.0 - probability).log()) / math.log(2.0)

    @staticmethod
    def _two_class_logits(foreground: torch.Tensor) -> torch.Tensor:
        # softmax([0, z]) is exactly sigmoid(z) for the foreground class.
        return torch.cat([torch.zeros_like(foreground), foreground], dim=1)

    def _run_rgb_branch(
        self,
        inputs: torch.Tensor,
        global_images: torch.Tensor | None = None,
        global_token: torch.Tensor | None = None,
    ) -> dict:
        rgb_features = self._encode_rgb(inputs, global_images, global_token)
        rgb_logits = self.rgb_aux_head(rgb_features[0])
        rgb_boundary_logits = self.rgb_boundary_head(rgb_features[0])
        decoded = self.decoder(rgb_features)
        zero_reliability = rgb_logits.new_zeros(rgb_logits.shape)
        refined = self.refiner(decoded, zero_reliability)
        return dict(
            rgb_features=rgb_features,
            rgb_logits=rgb_logits,
            rgb_boundary_logits=rgb_boundary_logits,
            final_logits=refined['final'],
            coarse_logits=refined['coarse'],
            boundary_logits=refined['boundary'],
            sdf=refined['sdf'],
            seg_logits=self._two_class_logits(refined['final']),
        )

    def _rectify_inputs(
        self,
        inputs: torch.Tensor,
        global_images: torch.Tensor | None = None,
        global_token: torch.Tensor | None = None,
    ) -> dict:
        _, depth, initial_reliability = self._split_inputs(inputs)
        rgb_features = self._encode_rgb(inputs, global_images, global_token)
        rgb_logits = self.rgb_aux_head(rgb_features[0])
        rgb_boundary_logits = self.rgb_boundary_head(rgb_features[0])
        rgb_boundary = rgb_boundary_logits.sigmoid()
        rectified = self.rectifier(
            depth, initial_reliability, rgb_features[0], rgb_boundary)
        return dict(
            rgb_features=rgb_features,
            rgb_logits=rgb_logits,
            rgb_boundary_logits=rgb_boundary_logits,
            rgb_boundary=rgb_boundary,
            rgb_uncertainty=self._entropy(rgb_logits),
            **rectified,
        )

    def _run_geometry_branch(
        self,
        inputs: torch.Tensor,
        global_images: torch.Tensor | None = None,
        global_token: torch.Tensor | None = None,
    ) -> dict:
        outputs = self._rectify_inputs(
            inputs, global_images=global_images, global_token=global_token)
        geometry_features = self.geometry_encoder(
            outputs['corrected_depth'], outputs['reliability'].detach())
        geometry_logits = self.geometry_aux_head(geometry_features[0])
        outputs.update(
            geometry_features=geometry_features,
            geometry_logits=geometry_logits,
            seg_logits=self._two_class_logits(geometry_logits),
        )
        return outputs

    def _run_network(
        self,
        inputs: torch.Tensor,
        global_images: torch.Tensor | None = None,
        global_token: torch.Tensor | None = None,
    ) -> dict:
        outputs = self._run_geometry_branch(
            inputs, global_images=global_images, global_token=global_token)
        rgb_features = outputs['rgb_features']
        geometry_features = outputs['geometry_features']
        # Reliability is calibrated by its own clean/corruption target.  Stop
        # segmentation gradients from turning it into a semantic mask; the
        # expert gates remain responsible for task-specific utility.
        reliability = outputs['reliability'].detach()
        disagreement = (
            outputs['rgb_logits'].sigmoid()
            - outputs['geometry_logits'].sigmoid()).abs()

        high_delta, low_delta = self.frequency_validator(
            rgb_features[0], geometry_features[0], reliability,
            outputs['rgb_boundary'], outputs['rgb_uncertainty'])
        fused_high, high_gate = self.high_fusion(
            rgb_features[0], geometry_features[0], high_delta,
            reliability, outputs['rgb_uncertainty'], disagreement)
        fused_low, low_gate = self.low_fusion(
            rgb_features[2], geometry_features[2], low_delta,
            reliability, outputs['rgb_uncertainty'], disagreement)
        fused_features = (
            fused_high, rgb_features[1], fused_low, rgb_features[3])
        decoded = self.decoder(fused_features)
        refined = self.refiner(decoded, reliability)
        outputs.update(
            final_logits=refined['final'],
            coarse_logits=refined['coarse'],
            boundary_logits=refined['boundary'],
            sdf=refined['sdf'],
            high_gate=high_gate,
            low_gate=low_gate,
            seg_logits=self._two_class_logits(refined['final']),
        )
        return outputs

    @staticmethod
    def _targets(
        data_samples: Sequence,
        size: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.stack([
            sample.gt_sem_seg.data for sample in data_samples
        ], dim=0).to(device=device)
        target = F.interpolate(target.float(), size=size, mode='nearest')
        valid = target != 255
        target = torch.where(valid, target, torch.zeros_like(target))
        return target, valid

    def _shape_losses(
        self,
        outputs: dict,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        boundary = _boundary_target(target, valid)
        boundary_loss = _binary_segmentation_loss(
            outputs['boundary_logits'], boundary, valid)
        boundary_loss += _binary_segmentation_loss(
            outputs['rgb_boundary_logits'], boundary, valid)
        sdf_target = _signed_distance_target(
            target, valid, truncation=self.sdf_truncation)
        sdf_error = F.smooth_l1_loss(
            outputs['sdf'].tanh(), sdf_target, reduction='none')
        return dict(
            loss_boundary=(
                self.loss_weights['boundary'] * boundary_loss * 0.5),
            loss_sdf=(
                self.loss_weights['sdf']
                * (sdf_error * valid.float()).sum()
                / valid.sum().clamp_min(1)),
        )

    def _equivariance_loss(
        self,
        inputs: torch.Tensor,
        global_token: torch.Tensor | None,
        reference_depth: torch.Tensor,
    ) -> torch.Tensor:
        # Alternate horizontal and vertical flips without introducing a third
        # interpolation of pseudo depth.
        dimension = -1 if torch.rand((), device=inputs.device) < 0.5 else -2
        flipped_inputs = torch.flip(inputs, dims=(dimension,))
        transformed = self._rectify_inputs(
            flipped_inputs, global_token=global_token)
        restored = torch.flip(
            transformed['corrected_depth'], dims=(dimension,))
        return F.smooth_l1_loss(restored, reference_depth.detach())

    def _geometry_losses(
        self,
        inputs: torch.Tensor,
        global_token: torch.Tensor | None,
        outputs: dict,
        target: torch.Tensor,
        valid: torch.Tensor,
        data_samples: Sequence,
    ) -> dict[str, torch.Tensor]:
        weights = self.loss_weights
        reliability_targets = []
        for sample in data_samples:
            if hasattr(sample, 'pseudo_validity'):
                reliability_targets.append(sample.pseudo_validity.data)
            else:
                reliability_targets.append(torch.ones_like(
                    outputs['learned_reliability'][0]))
        reliability_target = torch.stack(
            reliability_targets, dim=0).to(
                device=outputs['learned_reliability'].device,
                dtype=outputs['learned_reliability'].dtype)
        reliability_target = F.interpolate(
            reliability_target,
            size=outputs['learned_reliability'].shape[-2:],
            mode='bilinear', align_corners=False).clamp(0.0, 1.0)
        losses = dict(
            loss_geometry=(
                weights['geometry'] * _binary_segmentation_loss(
                    outputs['geometry_logits'], target, valid)),
            loss_reliability=(
                weights['reliability'] * F.binary_cross_entropy(
                    outputs['learned_reliability'].clamp(
                        1e-6, 1.0 - 1e-6),
                    reliability_target)),
            loss_preserve=(
                weights['preserve']
                * (outputs['corrected_depth']
                   - outputs['base_depth']).abs().mean()),
            loss_equivariance=(
                weights['equivariance'] * self._equivariance_loss(
                    inputs, global_token, outputs['corrected_depth'])),
        )
        losses['correction_scale'] = outputs['geometry_logits'].new_tensor(
            float(self.rectifier.correction_scale.detach()))
        return losses

    def _gate_losses(
        self,
        outputs: dict,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            rgb_error = F.binary_cross_entropy_with_logits(
                outputs['rgb_logits'], target, reduction='none')
            geometry_error = F.binary_cross_entropy_with_logits(
                outputs['geometry_logits'], target, reduction='none')
            advantage = (
                rgb_error - geometry_error - self.gate_margin).clamp_min(0.0)
            # No positive advantage means a closed residual branch.  This is
            # preferable to sigmoid(0)=0.5 for a fallback-first architecture.
            gate_target = 1.0 - torch.exp(
                -advantage / self.gate_temperature)
            boundary_band = F.max_pool2d(
                _boundary_target(target, valid), 5, stride=1, padding=2)
            reliability = outputs['reliability'].detach()
        high_gate = outputs['high_gate']
        high_reliability = F.interpolate(
            reliability, size=high_gate.shape[-2:], mode='bilinear',
            align_corners=False).expand_as(high_gate)
        high_effective = high_gate * high_reliability
        high_target = (
            gate_target * boundary_band).expand_as(high_gate) * high_reliability
        high_valid = valid.float().expand_as(high_gate)
        high_error = F.binary_cross_entropy(
            high_effective.clamp(1e-6, 1.0 - 1e-6), high_target,
            reduction='none')
        high_loss = (
            high_error * high_valid).sum() / high_valid.sum().clamp_min(1)
        low_gate = outputs['low_gate']
        low_target = F.interpolate(
            gate_target * (1.0 - boundary_band),
            size=low_gate.shape[-2:], mode='bilinear',
            align_corners=False).expand_as(low_gate)
        low_reliability = F.interpolate(
            reliability, size=low_gate.shape[-2:], mode='bilinear',
            align_corners=False).expand_as(low_gate)
        low_effective = low_gate * low_reliability
        low_target = low_target * low_reliability
        low_valid = F.interpolate(
            valid.float(), size=low_gate.shape[-2:],
            mode='nearest').expand_as(low_gate)
        low_error = F.binary_cross_entropy(
            low_effective.clamp(1e-6, 1.0 - 1e-6), low_target,
            reduction='none')
        low_loss = (
            low_error * low_valid).sum() / low_valid.sum().clamp_min(1)
        return dict(
            loss_gate=(
                self.loss_weights['gate'] * (high_loss + low_loss) * 0.5),
            high_gate_mean=high_gate.detach().mean(),
            low_gate_mean=low_gate.detach().mean(),
            high_effective_gate_mean=high_effective.detach().mean(),
            low_effective_gate_mean=low_effective.detach().mean(),
        )

    def _apply_geometry_dropout(self, inputs: torch.Tensor) -> torch.Tensor:
        """Train a true RGB fallback by disabling reliability per sample."""
        if (
            not self.training
            or self.training_stage != 'joint'
            or self.geometry_dropout_prob <= 0.0
        ):
            return inputs
        selected = torch.rand(
            inputs.shape[0], device=inputs.device) < self.geometry_dropout_prob
        if not selected.any():
            return inputs
        inputs = inputs.clone()
        inputs[selected, 4:5] = 0.0
        return inputs

    def loss(self, inputs: torch.Tensor, data_samples: Sequence) -> dict:
        global_images = self._global_images_from_samples(inputs, data_samples)
        global_token = (
            self._global_token(global_images)
            if self.use_global_context else None)
        inputs = self._apply_geometry_dropout(inputs)
        if self.training_stage == 'rgb':
            outputs = self._run_rgb_branch(
                inputs, global_token=global_token)
            target, valid = self._targets(
                data_samples, outputs['final_logits'].shape[-2:], inputs.device)
            losses = dict(
                loss_final=(
                    self.loss_weights['final'] * _binary_segmentation_loss(
                        outputs['final_logits'], target, valid)),
                loss_rgb=(
                    self.loss_weights['rgb'] * _binary_segmentation_loss(
                        outputs['rgb_logits'], target, valid)),
            )
            losses.update(self._shape_losses(outputs, target, valid))
            return losses

        if self.training_stage == 'geometry':
            outputs = self._run_geometry_branch(
                inputs, global_token=global_token)
            target, valid = self._targets(
                data_samples,
                outputs['geometry_logits'].shape[-2:], inputs.device)
            return self._geometry_losses(
                inputs, global_token, outputs, target, valid, data_samples)

        outputs = self._run_network(inputs, global_token=global_token)
        target, valid = self._targets(
            data_samples, outputs['final_logits'].shape[-2:], inputs.device)
        losses = dict(
            loss_final=(
                self.loss_weights['final'] * _binary_segmentation_loss(
                    outputs['final_logits'], target, valid)),
            loss_rgb=(
                self.loss_weights['rgb'] * _binary_segmentation_loss(
                    outputs['rgb_logits'], target, valid)),
        )
        losses.update(self._geometry_losses(
            inputs, global_token, outputs, target, valid, data_samples))
        losses.update(self._shape_losses(outputs, target, valid))
        losses.update(self._gate_losses(outputs, target, valid))
        return losses

    def encode_decode(
        self,
        inputs: torch.Tensor,
        batch_img_metas: list[dict] | None = None,
        global_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del batch_img_metas
        if self.training_stage == 'rgb':
            outputs = self._run_rgb_branch(inputs, global_token=global_token)
        elif self.training_stage == 'geometry':
            outputs = self._run_geometry_branch(inputs, global_token=global_token)
        else:
            outputs = self._run_network(inputs, global_token=global_token)
        logits = outputs['seg_logits']
        return F.interpolate(
            logits, size=inputs.shape[-2:], mode='bilinear',
            align_corners=False)

    def _forward(
        self, inputs: torch.Tensor, data_samples: Sequence | None = None
    ) -> torch.Tensor:
        del data_samples
        return self.encode_decode(inputs)

    def whole_inference(
        self, inputs: torch.Tensor, batch_img_metas: list[dict]
    ) -> torch.Tensor:
        global_token = None
        if self.use_global_context:
            global_images = F.interpolate(
                inputs[:, :3].float(),
                size=(self.global_thumbnail_size, self.global_thumbnail_size),
                mode='bilinear', align_corners=False)
            global_token = self._global_token(global_images)
        return self.encode_decode(
            inputs, batch_img_metas, global_token=global_token)

    def slide_inference(
        self, inputs: torch.Tensor, batch_img_metas: list[dict]
    ) -> torch.Tensor:
        h_stride, w_stride = self.test_cfg['stride']
        h_crop, w_crop = self.test_cfg['crop_size']
        batch, _, height, width = inputs.shape
        h_grids = max(height - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(width - w_crop + w_stride - 1, 0) // w_stride + 1
        predictions = inputs.new_zeros(batch, 2, height, width)
        counts = inputs.new_zeros(batch, 1, height, width)
        blend_windows: dict[tuple[int, int], torch.Tensor] = {}
        global_token = None
        if self.use_global_context:
            global_images = F.interpolate(
                inputs[:, :3].float(),
                size=(self.global_thumbnail_size, self.global_thumbnail_size),
                mode='bilinear', align_corners=False)
            global_token = self._global_token(global_images)

        for h_index in range(h_grids):
            for w_index in range(w_grids):
                y1 = h_index * h_stride
                x1 = w_index * w_stride
                y2 = min(y1 + h_crop, height)
                x2 = min(x1 + w_crop, width)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop = inputs[:, :, y1:y2, x1:x2]
                crop_metas = copy.deepcopy(batch_img_metas)
                for metadata in crop_metas:
                    metadata['img_shape'] = crop.shape[-2:]
                crop_logits = self.encode_decode(
                    crop, crop_metas, global_token=global_token)
                crop_size = crop_logits.shape[-2:]
                if crop_size not in blend_windows:
                    window_y = torch.hann_window(
                        crop_size[0], periodic=False,
                        device=crop_logits.device, dtype=crop_logits.dtype)
                    window_x = torch.hann_window(
                        crop_size[1], periodic=False,
                        device=crop_logits.device, dtype=crop_logits.dtype)
                    blend_windows[crop_size] = torch.outer(
                        window_y, window_x).clamp_min(0.05)[None, None]
                window = blend_windows[crop_size]
                predictions[:, :, y1:y2, x1:x2] += crop_logits * window
                counts[:, :, y1:y2, x1:x2] += window
        if (counts == 0).any():
            raise RuntimeError('slide inference left pixels uncovered')
        return predictions / counts

    def inference(
        self, inputs: torch.Tensor, batch_img_metas: list[dict]
    ) -> torch.Tensor:
        mode = self.test_cfg.get('mode', 'whole')
        if mode == 'slide':
            return self.slide_inference(inputs, batch_img_metas)
        if mode == 'whole':
            return self.whole_inference(inputs, batch_img_metas)
        raise ValueError(f'unsupported test mode: {mode}')

    def predict(
        self, inputs: torch.Tensor, data_samples: Sequence | None = None
    ) -> list:
        if data_samples is not None:
            batch_img_metas = [sample.metainfo for sample in data_samples]
        else:
            height, width = inputs.shape[-2:]
            batch_img_metas = [dict(
                ori_shape=(height, width),
                img_shape=(height, width),
                pad_shape=(height, width),
                padding_size=[0, 0, 0, 0],
            ) for _ in range(inputs.shape[0])]
        logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(logits, data_samples)
