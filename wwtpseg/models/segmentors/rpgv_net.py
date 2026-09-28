"""RPGV-Net: reliability-aware pseudo-geometry segmentation model."""

from __future__ import annotations

import copy
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
    GeometryFeaturePyramid,
    GlobalContextFiLM,
    HighResolutionDetailRefiner,
    MultiScaleDecoder,
    ReliabilityGuidedRectifier,
    ReliabilityWeightedResidualFusion,
    binary_entropy_from_logits,
)


def _binary_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Masked BCE + soft Dice for a one-logit binary prediction."""
    # Explicitly keep probabilities and full-resolution reductions in FP32;
    # do not depend on implicit promotion by the floating-point valid mask.
    logits = logits.float()
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

    DEFAULT_COMPONENTS = dict(
        global_context=True,
        depth_rectification=True,
        learned_reliability=True,
        frequency_validation=True,
        boundary_fusion=True,
        region_fusion=True,
        reliability_weighting=True,
        boundary_refinement=True,
        detail_refinement=True,
    )

    def __init__(
        self,
        rgb_encoder: dict,
        rgb_channels: Sequence[int] = (64, 128, 320, 512),
        geometry_channels: Sequence[int] = (32, 64, 128, 256),
        validation_channels: int = 32,
        decoder_channels: int = 128,
        detail_channels: int = 32,
        correction_scale: float = 0.1,
        max_correction_scale: float = 0.25,
        max_offset: float = 2.0,
        sdf_truncation: float = 5.0,
        geometry_dropout_prob: float = 0.15,
        training_stage: str = 'joint',
        use_global_context: bool = True,
        global_thumbnail_size: int = 512,
        component_cfg: dict[str, bool] | None = None,
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
        if sdf_truncation <= 0.0:
            raise ValueError('sdf_truncation must be positive')
        if not 0.0 <= geometry_dropout_prob <= 1.0:
            raise ValueError('geometry_dropout_prob must lie in [0, 1]')
        self.rgb_channels = tuple(rgb_channels)
        self.geometry_channels = tuple(geometry_channels)
        self.training_stage = training_stage
        self.component_cfg = self._resolve_component_cfg(
            use_global_context, component_cfg)
        # Keep this public attribute for compatibility with existing tooling
        # and checkpoints created before component_cfg was introduced.
        self.use_global_context = self.component_cfg['global_context']
        self.global_thumbnail_size = global_thumbnail_size
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or dict(mode='whole')
        self.align_corners = False
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
        self.geometry_pyramid = GeometryFeaturePyramid(geometry_channels)
        self.geometry_aux_head = nn.Conv2d(geometry_channels[0], 1, 1)
        self.frequency_validator = DualFrequencyGeometryValidator(
            rgb_channels[0], geometry_channels[0],
            rgb_channels[2], geometry_channels[2], validation_channels)
        self.high_fusion = ReliabilityWeightedResidualFusion(
            rgb_channels[0], validation_channels)
        self.low_fusion = ReliabilityWeightedResidualFusion(
            rgb_channels[2], validation_channels)
        self._build_prediction_modules(decoder_channels, detail_channels)

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

    def _build_prediction_modules(self, decoder_channels, detail_channels):
        """Construction hook: v2 replaces the prediction stack, not its weights."""
        self.decoder = MultiScaleDecoder(self.rgb_channels, decoder_channels)
        self.refiner = BoundaryResidualRefiner(decoder_channels)
        self.detail_refiner = HighResolutionDetailRefiner(
            decoder_channels, detail_channels)

    @classmethod
    def _resolve_component_cfg(
        cls,
        use_global_context: bool,
        component_cfg: dict[str, bool] | None,
    ) -> dict[str, bool]:
        """Validate and complete the configuration-driven component switches."""
        resolved = dict(cls.DEFAULT_COMPONENTS)
        resolved['global_context'] = use_global_context
        if component_cfg:
            unknown = set(component_cfg) - set(resolved)
            if unknown:
                raise ValueError(
                    f'unknown RPGV component switches: {sorted(unknown)}')
            invalid = {
                name: value for name, value in component_cfg.items()
                if not isinstance(value, bool)
            }
            if invalid:
                raise TypeError(
                    'RPGV component switches must be bool values; received '
                    f'{invalid}')
            resolved.update(component_cfg)
        return resolved

    def component_enabled(self, name: str) -> bool:
        """Return a validated component switch for tests and extensions."""
        return self.component_cfg[name]

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
                self.detail_refiner,
            ):
                self._set_trainable(module, True)
        elif self.training_stage == 'geometry':
            for module in (
                self.rectifier,
                self.geometry_encoder,
                self.geometry_pyramid,
                self.geometry_aux_head,
            ):
                self._set_trainable(module, True)

        # Disabled modules remain instantiated so all ablations share an
        # identical state_dict.  Freezing paths that are bypassed also makes
        # the variants safe with distributed training and avoids misleading
        # trainable-parameter counts.
        components = self.component_cfg
        if not components['global_context']:
            self._set_trainable(self.global_film, False)
        if not components['depth_rectification']:
            self.rectifier.correction_logit.requires_grad = False
            self._set_trainable(self.rectifier.residual, False)
        if not components['learned_reliability']:
            self._set_trainable(self.rectifier.learned_reliability, False)
        if (
            not components['depth_rectification']
            and not components['learned_reliability']
        ):
            self._set_trainable(self.rectifier, False)

        if not components['boundary_fusion']:
            self._set_trainable(self.high_fusion, False)
            for module in (
                self.frequency_validator.rgb_projection,
                self.frequency_validator.geometry_projection,
                self.frequency_validator.high_validator,
                self.frequency_validator.high_projection,
            ):
                self._set_trainable(module, False)
        elif not components['frequency_validation']:
            for module in (
                self.frequency_validator.rgb_projection,
                self.frequency_validator.high_validator,
                self.frequency_validator.high_projection,
            ):
                self._set_trainable(module, False)

        if not components['region_fusion']:
            self._set_trainable(self.low_fusion, False)
            for module in (
                self.frequency_validator.region_rgb_projection,
                self.frequency_validator.region_geometry_projection,
                self.frequency_validator.region_validator,
                self.frequency_validator.low_projection,
            ):
                self._set_trainable(module, False)
        elif not components['frequency_validation']:
            for module in (
                self.frequency_validator.region_rgb_projection,
                self.frequency_validator.region_validator,
                self.frequency_validator.low_projection,
            ):
                self._set_trainable(module, False)

        if not components['boundary_refinement']:
            self._set_trainable(self.refiner.boundary_gate, False)
            self._set_trainable(self.refiner.refinement, False)
        if not components['detail_refinement']:
            self._set_trainable(self.detail_refiner, False)
        if (
            self.training_stage == 'rgb'
            and self.loss_weights['boundary'] <= 0.0
        ):
            # The RGB boundary prediction only reaches the stage-one loss via
            # explicit boundary supervision.  Freeze it in the corresponding
            # loss ablation so DDP does not see an unused trainable branch.
            self._set_trainable(self.rgb_boundary_head, False)

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
                self.detail_refiner,
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
        return binary_entropy_from_logits(logits)

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
        rgb, _, _ = self._split_inputs(inputs)
        rgb_features = self._encode_rgb(inputs, global_images, global_token)
        rgb_logits = self.rgb_aux_head(rgb_features[0])
        rgb_boundary_logits = self.rgb_boundary_head(rgb_features[0])
        zero_reliability = rgb_logits.new_zeros(rgb_logits.shape)
        decoded_outputs = self._decode_and_refine(
            rgb, rgb_features, zero_reliability)
        return dict(
            rgb_features=rgb_features,
            rgb_logits=rgb_logits,
            rgb_boundary_logits=rgb_boundary_logits,
            **decoded_outputs,
        )

    def _decode_and_refine(
        self,
        rgb: torch.Tensor,
        features: Sequence[torch.Tensor],
        reliability: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """Decode features and apply only the enabled refinement stages."""
        decoded = self.decoder(features)
        refined = self.refiner(decoded, reliability)
        lowres_logits = (
            refined['final']
            if self.component_enabled('boundary_refinement')
            else refined['coarse'])
        detail_boundary = None
        if self.component_enabled('detail_refinement'):
            detail = self.detail_refiner(
                rgb, decoded, lowres_logits, reliability)
            final_logits = detail['final']
            detail_boundary = detail['boundary']
        else:
            final_logits = lowres_logits
        return dict(
            final_logits=final_logits,
            coarse_logits=refined['coarse'],
            boundary_logits=detail_boundary,
            lowres_boundary_logits=refined['boundary'],
            sdf=refined['sdf'],
            seg_logits=self._two_class_logits(final_logits),
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
            depth,
            initial_reliability,
            rgb_features[0],
            rgb_boundary,
            use_depth_correction=self.component_enabled(
                'depth_rectification'),
            use_learned_reliability=self.component_enabled(
                'learned_reliability'),
        )
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
        geometry_features = self.geometry_pyramid(geometry_features)
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
        # segmentation gradients from turning it into a semantic mask; DFGV
        # remains responsible for validating the candidate residual.
        reliability = outputs['reliability'].detach()
        validate_frequency = self.component_enabled('frequency_validation')
        weight_by_reliability = self.component_enabled(
            'reliability_weighting')

        fused_high = rgb_features[0]
        if self.component_enabled('boundary_fusion'):
            if validate_frequency:
                high_delta = self.frequency_validator.boundary_delta(
                    rgb_features[0], geometry_features[0], reliability,
                    outputs['rgb_boundary'])
            else:
                high_delta = self.frequency_validator.direct_boundary_delta(
                    geometry_features[0])
            fused_high = self.high_fusion(
                rgb_features[0], high_delta, reliability,
                use_reliability=weight_by_reliability)

        fused_low = rgb_features[2]
        if self.component_enabled('region_fusion'):
            if validate_frequency:
                low_delta = self.frequency_validator.region_delta(
                    rgb_features[2], geometry_features[2], reliability,
                    outputs['rgb_uncertainty'])
            else:
                low_delta = self.frequency_validator.direct_region_delta(
                    geometry_features[2])
            fused_low = self.low_fusion(
                rgb_features[2], low_delta, reliability,
                use_reliability=weight_by_reliability)
        fused_features = (
            fused_high, rgb_features[1], fused_low, rgb_features[3])
        rgb, _, _ = self._split_inputs(inputs)
        outputs.update(self._decode_and_refine(
            rgb, fused_features, reliability))
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

    @staticmethod
    def _resize_targets(
        target: torch.Tensor,
        valid: torch.Tensor,
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target.shape[-2:] == size:
            return target, valid
        target = F.interpolate(target.float(), size=size, mode='nearest')
        valid = F.interpolate(valid.float(), size=size, mode='nearest').bool()
        return target, valid

    def _shape_losses(
        self,
        outputs: dict,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        losses = {}
        if self.loss_weights['boundary'] > 0.0:
            boundary_predictions = [
                outputs['lowres_boundary_logits'],
                outputs['rgb_boundary_logits'],
            ]
            if outputs['boundary_logits'] is not None:
                boundary_predictions.insert(0, outputs['boundary_logits'])
            boundary_loss = target.new_zeros(())
            for prediction in boundary_predictions:
                boundary_target, boundary_valid = self._resize_targets(
                    target, valid, prediction.shape[-2:])
                boundary_target = _boundary_target(
                    boundary_target, boundary_valid)
                boundary_loss += _binary_segmentation_loss(
                    prediction, boundary_target, boundary_valid)
            losses['loss_boundary'] = (
                self.loss_weights['boundary']
                * boundary_loss / len(boundary_predictions))

        if self.loss_weights['sdf'] > 0.0:
            sdf_target_mask, sdf_valid = self._resize_targets(
                target, valid, outputs['sdf'].shape[-2:])
            sdf_target = _signed_distance_target(
                sdf_target_mask, sdf_valid, truncation=self.sdf_truncation)
            sdf_error = F.smooth_l1_loss(
                outputs['sdf'].float().tanh(), sdf_target.float(),
                reduction='none')
            losses['loss_sdf'] = (
                self.loss_weights['sdf']
                * (sdf_error * sdf_valid.float()).sum()
                / sdf_valid.sum().clamp_min(1))
        return losses

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
        return F.smooth_l1_loss(
            restored.float(), reference_depth.detach().float())

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
                dtype=torch.float32)
        reliability_target = F.interpolate(
            reliability_target,
            size=outputs['learned_reliability'].shape[-2:],
            mode='bilinear', align_corners=False).clamp(0.0, 1.0)
        losses = dict(
            loss_geometry=(
                weights['geometry'] * _binary_segmentation_loss(
                    outputs['geometry_logits'], target, valid)),
        )
        if (
            self.component_enabled('learned_reliability')
            and weights['reliability'] > 0.0
        ):
            losses['loss_reliability'] = (
                weights['reliability'] * F.binary_cross_entropy_with_logits(
                    outputs['learned_reliability_logits'].float(),
                    reliability_target))
        if self.component_enabled('depth_rectification'):
            if weights['preserve'] > 0.0:
                losses['loss_preserve'] = (
                    weights['preserve']
                    * (outputs['corrected_depth'].float()
                       - outputs['base_depth'].float()).abs().mean())
            if weights['equivariance'] > 0.0:
                losses['loss_equivariance'] = (
                    weights['equivariance'] * self._equivariance_loss(
                        inputs, global_token, outputs['corrected_depth']))
        losses['correction_scale'] = outputs['geometry_logits'].new_tensor(
            float(self.rectifier.correction_scale.detach()))
        return losses

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

    def parse_losses(self, losses: dict) -> tuple[torch.Tensor, dict]:
        """Stop before backward when a forward loss is already NaN/Inf.

        GradScaler still handles transient gradient overflow normally. It
        cannot repair a non-finite forward loss; repeatedly scaling that loss
        down can exhaust the scale and contaminate later checkpoints.
        """
        loss, log_vars = super().parse_losses(losses)
        if not torch.isfinite(loss.detach()).all():
            invalid = [
                name for name, value in log_vars.items()
                if not torch.isfinite(value.detach()).all()
            ]
            raise FloatingPointError(
                f'RPGV {self.training_stage}: non-finite losses: '
                f'{", ".join(invalid)}. Stopped before backward; check inputs '
                'and checkpoint weights. Do not resume a NaN checkpoint.')
        return loss, log_vars

    def loss(self, inputs: torch.Tensor, data_samples: Sequence) -> dict:
        global_token = None
        if self.use_global_context:
            global_images = self._global_images_from_samples(
                inputs, data_samples)
            global_token = self._global_token(global_images)
        inputs = self._apply_geometry_dropout(inputs)
        if self.training_stage == 'rgb':
            outputs = self._run_rgb_branch(
                inputs, global_token=global_token)
            target, valid = self._targets(
                data_samples, inputs.shape[-2:], inputs.device)
            final_logits = F.interpolate(
                outputs['final_logits'], size=target.shape[-2:],
                mode='bilinear', align_corners=False)
            rgb_target, rgb_valid = self._resize_targets(
                target, valid, outputs['rgb_logits'].shape[-2:])
            losses = dict(
                loss_final=(
                    self.loss_weights['final'] * _binary_segmentation_loss(
                        final_logits, target, valid)),
                loss_rgb=(
                    self.loss_weights['rgb'] * _binary_segmentation_loss(
                        outputs['rgb_logits'], rgb_target, rgb_valid)),
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
            data_samples, inputs.shape[-2:], inputs.device)
        final_logits = F.interpolate(
            outputs['final_logits'], size=target.shape[-2:],
            mode='bilinear', align_corners=False)
        rgb_target, rgb_valid = self._resize_targets(
            target, valid, outputs['rgb_logits'].shape[-2:])
        geometry_target, geometry_valid = self._resize_targets(
            target, valid, outputs['geometry_logits'].shape[-2:])
        losses = dict(
            loss_final=(
                self.loss_weights['final'] * _binary_segmentation_loss(
                    final_logits, target, valid)),
            loss_rgb=(
                self.loss_weights['rgb'] * _binary_segmentation_loss(
                    outputs['rgb_logits'], rgb_target, rgb_valid)),
        )
        losses.update(self._geometry_losses(
            inputs, global_token, outputs,
            geometry_target, geometry_valid, data_samples))
        losses.update(self._shape_losses(outputs, target, valid))
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
