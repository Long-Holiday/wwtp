"""RPGV full multi-class segmentation model with Depth-Anything geometry branch for remote sensing."""

from __future__ import annotations

import copy
import math
from typing import Sequence

import torch
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from torch import nn
from torch.nn import functional as F

from ..models.utils.rpgv_modules import (
    ConvNormAct,
    DepthwiseSeparableBlock,
    DualFrequencyGeometryValidator,
    GeometryEncoder,
    GeometryFeaturePyramid,
    GlobalContextFiLM,
    MultiScaleDecoder,
    ReliabilityGuidedRectifier,
    ReliabilityWeightedResidualFusion,
)


class MultiClassBoundaryResidualRefinerFull(nn.Module):
    """Multi-class boundary-guided residual refiner at stride 4 with reliability gating."""

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.coarse_head = nn.Conv2d(channels, num_classes, 1)
        self.boundary_head = nn.Sequential(
            ConvNormAct(channels, max(channels // 2, 16)),
            nn.Conv2d(max(channels // 2, 16), 1, 1),
        )
        # Gate takes: boundary (1) + uncertainty (1) + reliability (1) = 3 channels
        self.boundary_gate = nn.Sequential(
            ConvNormAct(3, 16),
            nn.Conv2d(16, 1, 1),
        )
        # Refinement takes: feature (C) + coarse (num_classes) + boundary (1) + reliability (1)
        self.refinement = nn.Sequential(
            ConvNormAct(channels + num_classes + 2, channels),
            nn.Conv2d(channels, num_classes, 1),
        )
        # Identity initialization: zero-init residual projection and reliability gate slice
        nn.init.zeros_(self.boundary_gate[0][0].weight[:, 2:3])
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)

    def forward(
        self, feature: torch.Tensor, reliability: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        coarse = self.coarse_head(feature)
        boundary = self.boundary_head(feature)
        probs = coarse.softmax(dim=1)
        log_c = math.log(max(self.num_classes, 2))
        uncertainty = -(probs * (probs + 1e-7).log()).sum(dim=1, keepdim=True) / log_c

        if reliability is not None:
            rel = F.interpolate(
                reliability, size=feature.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            rel = torch.zeros_like(uncertainty)

        gate = self.boundary_gate(
            torch.cat([boundary.sigmoid(), uncertainty, rel], dim=1)
        ).sigmoid()

        residual = self.refinement(
            torch.cat([feature, coarse, boundary, rel], dim=1)
        )
        final = coarse + gate * residual
        return dict(
            final=final,
            coarse=coarse,
            boundary=boundary,
            refinement_gate=gate,
        )


class MultiClassDetailRefinerFull(nn.Module):
    """High-resolution detail refiner recovering stride-2 spatial details with reliability gating."""

    def __init__(
        self, decoder_channels: int, num_classes: int, channels: int = 32
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.rgb_stem = nn.Sequential(
            ConvNormAct(3, channels, stride=2),
            DepthwiseSeparableBlock(channels, channels),
        )
        self.decoder_projection = ConvNormAct(
            decoder_channels, channels, kernel_size=1
        )
        self.fuse = nn.Sequential(
            ConvNormAct(2 * channels + num_classes, channels),
            DepthwiseSeparableBlock(channels, channels),
        )
        self.boundary_head = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, 1, 1),
        )
        # Gate takes: boundary (1) + uncertainty (1) + reliability (1) = 3 channels
        self.gate = nn.Sequential(
            ConvNormAct(3, 16),
            nn.Conv2d(16, 1, 1),
        )
        self.residual = nn.Sequential(
            ConvNormAct(channels + num_classes + 1, channels),
            nn.Conv2d(channels, num_classes, 1),
        )
        # Identity initialization
        nn.init.zeros_(self.gate[0][0].weight[:, 2:3])
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        rgb: torch.Tensor,
        decoder_feature: torch.Tensor,
        base_logits: torch.Tensor,
        reliability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        rgb_detail = self.rgb_stem(rgb)
        size = rgb_detail.shape[-2:]
        decoder_detail = F.interpolate(
            self.decoder_projection(decoder_feature),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        base_logits_up = F.interpolate(
            base_logits, size=size, mode="bilinear", align_corners=False
        )
        probs_up = base_logits_up.softmax(dim=1)
        detail = self.fuse(
            torch.cat([rgb_detail, decoder_detail, probs_up], dim=1)
        )
        boundary = self.boundary_head(detail)
        log_c = math.log(max(self.num_classes, 2))
        uncertainty = -(probs_up * (probs_up + 1e-7).log()).sum(
            dim=1, keepdim=True
        ) / log_c

        if reliability is not None:
            rel = F.interpolate(
                reliability, size=size, mode="bilinear", align_corners=False
            )
        else:
            rel = torch.zeros_like(uncertainty)

        gate = self.gate(
            torch.cat([boundary.sigmoid(), uncertainty, rel], dim=1)
        ).sigmoid()
        residual = self.residual(
            torch.cat([detail, base_logits_up, rel], dim=1)
        )
        final = base_logits_up + gate * residual
        return dict(
            final=final,
            boundary=boundary,
            refinement_gate=gate,
        )


@MODELS.register_module()
class RPGVRemoteSensingFull(BaseSegmentor):
    """Complete RPGV segmentation model for multi-class remote sensing with Depth-Anything geometry branch.

    Integrates:
    1. Pretrained hierarchical transformer backbone (MiT-B2)
    2. GlobalContextFiLM modulating multi-scale feature hierarchies via whole-scene tokens
    3. ReliabilityGuidedRectifier (RGR) correcting pseudo depth via deformable convolution and predicting learned reliability
    4. GeometryEncoder & GeometryFeaturePyramid for multi-scale geometry representations
    5. DualFrequencyGeometryValidator (DFGV) using 2D Haar wavelets to validate boundary (HF) and region (LF) residuals
    6. ReliabilityWeightedResidualFusion (RWRF) injecting validated residuals into Stage 0 and Stage 2
    7. MultiScaleDecoder multi-level feature aggregation
    8. MultiClassBoundaryResidualRefinerFull and MultiClassDetailRefinerFull
    9. Multi-task supervision: final, coarse, rgb_aux, geometry_aux, boundary, preserve, equivariance, reliability.
    """

    def __init__(
        self,
        rgb_encoder: dict,
        num_classes: int,
        ignore_index: int = 255,
        rgb_channels: Sequence[int] = (64, 128, 320, 512),
        geometry_channels: Sequence[int] = (32, 64, 128, 256),
        validation_channels: int = 32,
        decoder_channels: int = 256,
        detail_channels: int = 32,
        correction_scale: float = 0.1,
        max_correction_scale: float = 0.25,
        max_offset: float = 2.0,
        geometry_dropout_prob: float = 0.15,
        use_global_context: bool = True,
        global_thumbnail_size: int = 512,
        loss_weights: dict | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg
        )
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.rgb_channels = tuple(rgb_channels)
        self.geometry_channels = tuple(geometry_channels)
        self.validation_channels = validation_channels
        self.decoder_channels = decoder_channels
        self.detail_channels = detail_channels
        self.geometry_dropout_prob = geometry_dropout_prob
        self.use_global_context = use_global_context
        self.global_thumbnail_size = global_thumbnail_size
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or dict(mode="whole")
        self.align_corners = False

        self.loss_weights = dict(
            final=1.0,
            coarse=0.4,
            aux=0.2,
            geometry=0.2,
            boundary=0.2,
            preserve=0.05,
            equivariance=0.05,
            reliability=0.05,
        )
        if loss_weights:
            self.loss_weights.update(loss_weights)

        # 1. RGB Encoder
        self.rgb_encoder = MODELS.build(rgb_encoder)

        # 2. Global Context FiLM
        self.global_film = GlobalContextFiLM(
            rgb_channels[-1], rgb_channels
        )

        # 3. Aux Head at stride 4 (RGB branch)
        self.aux_head = nn.Conv2d(rgb_channels[0], num_classes, 1)

        # 4. RGB Boundary Head at stride 4
        self.rgb_boundary_head = nn.Sequential(
            nn.Conv2d(rgb_channels[0], validation_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(validation_channels, 1, 1),
        )

        # 5. Reliability Guided Rectifier (RGR)
        self.rectifier = ReliabilityGuidedRectifier(
            rgb_channels=rgb_channels[0],
            channels=geometry_channels[0],
            max_offset=max_offset,
            correction_scale=correction_scale,
            max_correction_scale=max_correction_scale,
        )

        # 6. Geometry Encoder & Feature Pyramid
        self.geometry_encoder = GeometryEncoder(geometry_channels)
        self.geometry_pyramid = GeometryFeaturePyramid(geometry_channels)
        self.geometry_aux_head = nn.Conv2d(geometry_channels[0], num_classes, 1)

        # 7. Dual Frequency Geometry Validator (DFGV)
        self.frequency_validator = DualFrequencyGeometryValidator(
            rgb_channels[0],
            geometry_channels[0],
            rgb_channels[2],
            geometry_channels[2],
            validation_channels,
        )

        # 8. Reliability Weighted Residual Fusion (RWRF)
        self.high_fusion = ReliabilityWeightedResidualFusion(
            rgb_channels[0], validation_channels
        )
        self.low_fusion = ReliabilityWeightedResidualFusion(
            rgb_channels[2], validation_channels
        )

        # 9. Multi-Scale Decoder
        self.decoder = MultiScaleDecoder(rgb_channels, decoder_channels)

        # 10. Stride-4 Boundary Residual Refiner
        self.refiner = MultiClassBoundaryResidualRefinerFull(
            decoder_channels, num_classes
        )

        # 11. Stride-2 High-Resolution Detail Refiner
        self.detail_refiner = MultiClassDetailRefinerFull(
            decoder_channels, num_classes, detail_channels
        )

        # ImageNet RGB normalization buffers
        self.register_buffer(
            "rgb_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _normalize_bgr(self, images: torch.Tensor) -> torch.Tensor:
        # Flip BGR to RGB, then normalize with ImageNet mean & std
        images = images.float().flip(1)
        return (images - self.rgb_mean) / self.rgb_std

    def _split_inputs(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = inputs.float()
        rgb = self._normalize_bgr(inputs[:, :3])
        if inputs.shape[1] >= 5:
            depth = (inputs[:, 3:4] / 255.0).clamp(0.0, 1.0)
            reliability = (inputs[:, 4:5] / 255.0).clamp(0.0, 1.0)
        else:
            depth = inputs.new_zeros(inputs.shape[0], 1, *inputs.shape[-2:])
            reliability = torch.zeros_like(depth)
        return rgb, depth, reliability

    def _global_token(self, global_images: torch.Tensor) -> torch.Tensor:
        norm_images = self._normalize_bgr(global_images)
        global_features = self.rgb_encoder(norm_images)
        return global_features[-1].mean(dim=(2, 3))

    def _global_images_from_samples(
        self, inputs: torch.Tensor, data_samples: Sequence | None
    ) -> torch.Tensor:
        if data_samples and all(
            hasattr(sample, "global_img") for sample in data_samples
        ):
            return torch.stack(
                [sample.global_img.data for sample in data_samples], dim=0
            ).to(device=inputs.device, dtype=inputs.dtype)
        return F.interpolate(
            inputs[:, :3].float(),
            size=(self.global_thumbnail_size, self.global_thumbnail_size),
            mode="bilinear",
            align_corners=False,
        )

    def extract_feat(
        self, inputs: torch.Tensor, global_token: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        rgb, _, _ = self._split_inputs(inputs)
        features = tuple(self.rgb_encoder(rgb))
        if self.use_global_context:
            if global_token is None:
                global_images = F.interpolate(
                    inputs[:, :3].float(),
                    size=(
                        self.global_thumbnail_size,
                        self.global_thumbnail_size,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
                global_token = self._global_token(global_images)
            features = self.global_film(features, global_token)
        return features

    def _apply_geometry_dropout(self, inputs: torch.Tensor) -> torch.Tensor:
        """Randomly zero out reliability during training to preserve robust RGB fallback."""
        if not self.training or self.geometry_dropout_prob <= 0.0 or inputs.shape[1] < 5:
            return inputs
        selected = (
            torch.rand(inputs.shape[0], device=inputs.device)
            < self.geometry_dropout_prob
        )
        if not selected.any():
            return inputs
        inputs = inputs.clone()
        inputs[selected, 4:5] = 0.0
        return inputs

    def _rectify_inputs(
        self,
        inputs: torch.Tensor,
        global_token: torch.Tensor | None = None,
    ) -> dict:
        rgb, depth, initial_reliability = self._split_inputs(inputs)
        rgb_features = tuple(self.rgb_encoder(rgb))
        if self.use_global_context and global_token is not None:
            rgb_features = self.global_film(rgb_features, global_token)

        rgb_logits = self.aux_head(rgb_features[0])
        rgb_boundary_logits = self.rgb_boundary_head(rgb_features[0])
        rgb_boundary = rgb_boundary_logits.sigmoid()

        # Multi-class entropy uncertainty for region validation
        probs = rgb_logits.softmax(dim=1)
        log_c = math.log(max(self.num_classes, 2))
        rgb_uncertainty = -(probs * (probs + 1e-7).log()).sum(
            dim=1, keepdim=True
        ) / log_c

        rectified = self.rectifier(
            depth,
            initial_reliability,
            rgb_features[0],
            rgb_boundary,
            use_depth_correction=True,
            use_learned_reliability=True,
        )
        return dict(
            rgb=rgb,
            depth=depth,
            rgb_features=rgb_features,
            rgb_logits=rgb_logits,
            rgb_boundary_logits=rgb_boundary_logits,
            rgb_boundary=rgb_boundary,
            rgb_uncertainty=rgb_uncertainty,
            **rectified,
        )

    def _run_network(
        self,
        inputs: torch.Tensor,
        global_token: torch.Tensor | None = None,
    ) -> dict:
        rect = self._rectify_inputs(inputs, global_token=global_token)
        rgb_features = rect["rgb_features"]
        reliability = rect["reliability"].detach()

        # 1. Encode geometry features
        geometry_features = self.geometry_encoder(
            rect["corrected_depth"], reliability
        )
        geometry_features = self.geometry_pyramid(geometry_features)
        geometry_logits = self.geometry_aux_head(geometry_features[0])

        # 2. Dual-frequency validation & residual fusion
        high_delta = self.frequency_validator.boundary_delta(
            rgb_features[0],
            geometry_features[0],
            reliability,
            rect["rgb_boundary"],
        )
        fused_high = self.high_fusion(
            rgb_features[0], high_delta, reliability, use_reliability=True
        )

        low_delta = self.frequency_validator.region_delta(
            rgb_features[2],
            geometry_features[2],
            reliability,
            rect["rgb_uncertainty"],
        )
        fused_low = self.low_fusion(
            rgb_features[2], low_delta, reliability, use_reliability=True
        )

        fused_features = (
            fused_high,
            rgb_features[1],
            fused_low,
            rgb_features[3],
        )

        # 3. Multi-scale decoding
        decoded_features = self.decoder(fused_features)

        # 4. Refinement
        refined = self.refiner(decoded_features, reliability=reliability)
        detail = self.detail_refiner(
            inputs[:, :3],
            decoded_features,
            refined["final"],
            reliability=reliability,
        )

        boundary_predictions = [
            detail["boundary"],
            refined["boundary"],
            rect["rgb_boundary_logits"],
        ]

        return dict(
            final_logits=detail["final"],
            coarse_logits=refined["coarse"],
            aux_logits=rect["rgb_logits"],
            geometry_logits=geometry_logits,
            boundary_predictions=boundary_predictions,
            refinement_gate=detail["refinement_gate"],
            rectified=rect,
        )

    @staticmethod
    def _extract_boundary(
        target: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract class-agnostic semantic boundary labels from multi-class ground truth."""
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        target_f = target.float()
        valid_f = valid.float()

        pad_t = F.pad(target_f, (1, 1, 1, 1), mode="replicate")
        pad_v = F.pad(valid_f, (1, 1, 1, 1), mode="replicate")

        up_t, up_v = pad_t[:, :, :-2, 1:-1], pad_v[:, :, :-2, 1:-1]
        down_t, down_v = pad_t[:, :, 2:, 1:-1], pad_v[:, :, 2:, 1:-1]
        left_t, left_v = pad_t[:, :, 1:-1, :-2], pad_v[:, :, 1:-1, :-2]
        right_t, right_v = pad_t[:, :, 1:-1, 2:], pad_v[:, :, 1:-1, 2:]

        boundary = (
            ((target_f != up_t) & (valid_f > 0.5) & (up_v > 0.5))
            | ((target_f != down_t) & (valid_f > 0.5) & (down_v > 0.5))
            | ((target_f != left_t) & (valid_f > 0.5) & (left_v > 0.5))
            | ((target_f != right_t) & (valid_f > 0.5) & (right_v > 0.5))
        ).float()
        return boundary, valid_f

    def _equivariance_loss(
        self,
        inputs: torch.Tensor,
        global_token: torch.Tensor | None,
        reference_depth: torch.Tensor,
    ) -> torch.Tensor:
        dimension = -1 if torch.rand((), device=inputs.device) < 0.5 else -2
        flipped_inputs = torch.flip(inputs, dims=(dimension,))
        transformed = self._rectify_inputs(
            flipped_inputs, global_token=global_token
        )
        restored = torch.flip(
            transformed["corrected_depth"], dims=(dimension,)
        )
        return F.smooth_l1_loss(
            restored.float(), reference_depth.detach().float()
        )

    def loss(self, inputs: torch.Tensor, data_samples: Sequence) -> dict:
        inputs = self._apply_geometry_dropout(inputs)

        global_token = None
        if self.use_global_context:
            global_images = self._global_images_from_samples(
                inputs, data_samples
            )
            global_token = self._global_token(global_images)

        outputs = self._run_network(inputs, global_token=global_token)
        rect = outputs["rectified"]

        # Ground truth targets
        targets = torch.stack(
            [sample.gt_sem_seg.data for sample in data_samples], dim=0
        ).to(device=inputs.device)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        targets = targets.long()

        valid = targets != self.ignore_index
        if not valid.any():
            return dict(
                loss_final=outputs["final_logits"].sum() * 0.0,
                loss_coarse=outputs["coarse_logits"].sum() * 0.0,
                loss_aux=outputs["aux_logits"].sum() * 0.0,
                loss_geometry=outputs["geometry_logits"].sum() * 0.0,
                loss_boundary=outputs["boundary_predictions"][0].sum() * 0.0,
            )

        losses = {}

        # 1. Final Loss (stride 2 upsampled to full resolution)
        final_logits = F.interpolate(
            outputs["final_logits"],
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        losses["loss_final"] = self.loss_weights["final"] * F.cross_entropy(
            final_logits, targets, ignore_index=self.ignore_index
        )

        # 2. Coarse Loss (stride 4)
        targets_s4 = F.interpolate(
            targets.unsqueeze(1).float(),
            size=outputs["coarse_logits"].shape[-2:],
            mode="nearest",
        ).squeeze(1).long()
        losses["loss_coarse"] = self.loss_weights["coarse"] * F.cross_entropy(
            outputs["coarse_logits"],
            targets_s4,
            ignore_index=self.ignore_index,
        )

        # 3. Aux Loss (RGB stride 4)
        losses["loss_aux"] = self.loss_weights["aux"] * F.cross_entropy(
            outputs["aux_logits"],
            targets_s4,
            ignore_index=self.ignore_index,
        )

        # 4. Geometry Aux Loss (stride 4)
        losses["loss_geometry"] = self.loss_weights["geometry"] * F.cross_entropy(
            outputs["geometry_logits"],
            targets_s4,
            ignore_index=self.ignore_index,
        )

        # 5. Boundary Loss
        boundary_target_full, boundary_valid_full = self._extract_boundary(
            targets, valid
        )
        boundary_losses = []
        for b_pred in outputs["boundary_predictions"]:
            b_tgt = F.interpolate(
                boundary_target_full,
                size=b_pred.shape[-2:],
                mode="nearest",
            )
            b_val = F.interpolate(
                boundary_valid_full,
                size=b_pred.shape[-2:],
                mode="nearest",
            )
            denom = b_val.sum().clamp_min(1.0)
            b_loss = (
                F.binary_cross_entropy_with_logits(
                    b_pred, b_tgt, reduction="none"
                )
                * b_val
            ).sum() / denom
            boundary_losses.append(b_loss)

        losses["loss_boundary"] = self.loss_weights["boundary"] * (
            sum(boundary_losses) / len(boundary_losses)
        )

        # 6. Geometry Regularization Losses
        if self.loss_weights.get("preserve", 0.0) > 0.0:
            losses["loss_preserve"] = self.loss_weights["preserve"] * (
                rect["corrected_depth"].float() - rect["base_depth"].float()
            ).abs().mean()

        if self.loss_weights.get("equivariance", 0.0) > 0.0:
            losses["loss_equivariance"] = self.loss_weights[
                "equivariance"
            ] * self._equivariance_loss(
                inputs, global_token, rect["corrected_depth"]
            )

        if self.loss_weights.get("reliability", 0.0) > 0.0:
            rel_targets = []
            for sample in data_samples:
                if hasattr(sample, "pseudo_validity"):
                    rel_targets.append(sample.pseudo_validity.data)
                else:
                    rel_targets.append(
                        torch.ones_like(rect["learned_reliability"][0])
                    )
            rel_target = torch.stack(rel_targets, dim=0).to(
                device=rect["learned_reliability"].device, dtype=torch.float32
            )
            rel_target = F.interpolate(
                rel_target,
                size=rect["learned_reliability"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            losses["loss_reliability"] = self.loss_weights[
                "reliability"
            ] * F.binary_cross_entropy_with_logits(
                rect["learned_reliability_logits"].float(), rel_target
            )

        return losses

    def encode_decode(
        self,
        inputs: torch.Tensor,
        batch_img_metas: list[dict] | None = None,
        global_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del batch_img_metas
        outputs = self._run_network(inputs, global_token=global_token)
        return F.interpolate(
            outputs["final_logits"],
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )

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
                mode="bilinear",
                align_corners=False,
            )
            global_token = self._global_token(global_images)
        return self.encode_decode(
            inputs, batch_img_metas, global_token=global_token
        )

    def slide_inference(
        self, inputs: torch.Tensor, batch_img_metas: list[dict]
    ) -> torch.Tensor:
        global_token = None
        if self.use_global_context:
            global_images = F.interpolate(
                inputs[:, :3].float(),
                size=(self.global_thumbnail_size, self.global_thumbnail_size),
                mode="bilinear",
                align_corners=False,
            )
            global_token = self._global_token(global_images)

        h_stride, w_stride = self.test_cfg.get("stride", (512, 512))
        h_crop, w_crop = self.test_cfg.get("crop_size", (512, 512))
        batch_size, _, h_img, w_img = inputs.size()
        num_classes = self.num_classes

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        predictions = inputs.new_zeros((batch_size, num_classes, h_img, w_img))
        counts = inputs.new_zeros((batch_size, 1, h_img, w_img))
        blend_windows = {}

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_inputs = inputs[:, :, y1:y2, x1:x2]
                crop_logits = self.encode_decode(
                    crop_inputs, batch_img_metas, global_token=global_token
                )

                crop_size = (y2 - y1, x2 - x1)
                if crop_size not in blend_windows:
                    window_y = torch.hann_window(
                        crop_size[0],
                        periodic=False,
                        device=crop_logits.device,
                        dtype=crop_logits.dtype,
                    )
                    window_x = torch.hann_window(
                        crop_size[1],
                        periodic=False,
                        device=crop_logits.device,
                        dtype=crop_logits.dtype,
                    )
                    blend_windows[crop_size] = torch.outer(
                        window_y, window_x
                    ).clamp_min(0.05)[None, None]
                window = blend_windows[crop_size]
                predictions[:, :, y1:y2, x1:x2] += crop_logits * window
                counts[:, :, y1:y2, x1:x2] += window

        if (counts == 0).any():
            raise RuntimeError("slide inference left pixels uncovered")
        return predictions / counts

    def inference(
        self, inputs: torch.Tensor, batch_img_metas: list[dict]
    ) -> torch.Tensor:
        mode = self.test_cfg.get("mode", "whole")
        if mode == "slide":
            return self.slide_inference(inputs, batch_img_metas)
        if mode == "whole":
            return self.whole_inference(inputs, batch_img_metas)
        raise ValueError(f"unsupported test mode: {mode}")

    def predict(
        self, inputs: torch.Tensor, data_samples: Sequence | None = None
    ) -> list:
        height, width = inputs.shape[-2:]
        if data_samples is not None:
            batch_img_metas = []
            for sample in data_samples:
                if "ori_shape" not in sample.metainfo:
                    sample.set_metainfo(dict(ori_shape=(height, width)))
                if "img_shape" not in sample.metainfo:
                    sample.set_metainfo(dict(img_shape=(height, width)))
                batch_img_metas.append(sample.metainfo)
        else:
            batch_img_metas = [
                dict(
                    ori_shape=(height, width),
                    img_shape=(height, width),
                    pad_shape=(height, width),
                    padding_size=[0, 0, 0, 0],
                )
                for _ in range(inputs.shape[0])
            ]
        logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(logits, data_samples)
