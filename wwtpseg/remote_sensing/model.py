"""RPGV multi-class segmentation model adapted for remote sensing benchmarks."""

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
    GlobalContextFiLM,
    MultiScaleDecoder,
)


class MultiClassBoundaryResidualRefiner(nn.Module):
    """Multi-class boundary-guided residual refiner at stride 4.

    Computes semantic boundaries between classes and estimates normalized
    classification uncertainty to gate residual logit corrections.
    """

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.coarse_head = nn.Conv2d(channels, num_classes, 1)
        self.boundary_head = nn.Sequential(
            ConvNormAct(channels, max(channels // 2, 16)),
            nn.Conv2d(max(channels // 2, 16), 1, 1),
        )
        self.boundary_gate = nn.Sequential(
            ConvNormAct(2, 16),
            nn.Conv2d(16, 1, 1),
        )
        self.refinement = nn.Sequential(
            ConvNormAct(channels + num_classes + 1, channels),
            nn.Conv2d(channels, num_classes, 1),
        )
        # Identity initialization: zero-init residual projection
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        coarse = self.coarse_head(feature)
        boundary = self.boundary_head(feature)
        probs = coarse.softmax(dim=1)
        log_c = math.log(max(self.num_classes, 2))
        uncertainty = -(probs * (probs + 1e-7).log()).sum(dim=1, keepdim=True) / log_c
        gate = self.boundary_gate(
            torch.cat([boundary.sigmoid(), uncertainty], dim=1)
        ).sigmoid()
        residual = self.refinement(
            torch.cat([feature, coarse, boundary], dim=1)
        )
        final = coarse + gate * residual
        return dict(
            final=final,
            coarse=coarse,
            boundary=boundary,
            refinement_gate=gate,
        )


class MultiClassDetailRefiner(nn.Module):
    """High-resolution detail refiner recovering stride-2 spatial details.

    Merges low-level shallow RGB stem features with upsampled decoder features
    and coarse multi-class probability maps.
    """

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
        self.gate = nn.Sequential(
            ConvNormAct(2, 16),
            nn.Conv2d(16, 1, 1),
        )
        self.residual = nn.Sequential(
            ConvNormAct(channels + num_classes, channels),
            nn.Conv2d(channels, num_classes, 1),
        )
        # Identity initialization: zero-init residual projection
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        rgb: torch.Tensor,
        decoder_feature: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rgb_detail = self.rgb_stem(rgb)
        size = rgb_detail.shape[-2:]
        decoder_detail = F.interpolate(
            self.decoder_projection(decoder_feature),
            size=size,
            mode='bilinear',
            align_corners=False,
        )
        base_logits_up = F.interpolate(
            base_logits, size=size, mode='bilinear', align_corners=False
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
        gate = self.gate(
            torch.cat([boundary.sigmoid(), uncertainty], dim=1)
        ).sigmoid()
        residual = self.residual(torch.cat([detail, base_logits_up], dim=1))
        final = base_logits_up + gate * residual
        return dict(
            final=final,
            boundary=boundary,
            refinement_gate=gate,
        )


@MODELS.register_module()
class RPGVRemoteSensing(BaseSegmentor):
    """RPGV segmentation model for multi-class remote sensing benchmarks.

    Adapts RPGV's core architectural strengths to optical remote sensing:
    1. Pretrained hierarchical transformer backbone (MiT-B2)
    2. GlobalContextFiLM modulating multi-scale feature hierarchies via whole-scene tokens
    3. MultiScaleDecoder multi-level feature aggregation
    4. Multi-class boundary residual refinement and stride-2 detail refinement
    """

    def __init__(
        self,
        rgb_encoder: dict,
        num_classes: int,
        ignore_index: int = 255,
        rgb_channels: Sequence[int] = (64, 128, 320, 512),
        decoder_channels: int = 256,
        detail_channels: int = 32,
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
        self.decoder_channels = decoder_channels
        self.detail_channels = detail_channels
        self.use_global_context = use_global_context
        self.global_thumbnail_size = global_thumbnail_size
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or dict(mode='whole')
        self.align_corners = False

        self.loss_weights = dict(
            final=1.0,
            coarse=0.4,
            aux=0.2,
            boundary=0.2,
        )
        if loss_weights:
            self.loss_weights.update(loss_weights)

        # 1. RGB Encoder
        self.rgb_encoder = MODELS.build(rgb_encoder)

        # 2. Global Context FiLM
        self.global_film = GlobalContextFiLM(
            rgb_channels[-1], rgb_channels
        )

        # 3. Aux Head at stride 4
        self.aux_head = nn.Conv2d(rgb_channels[0], num_classes, 1)

        # 4. Multi-Scale Decoder
        self.decoder = MultiScaleDecoder(rgb_channels, decoder_channels)

        # 5. Stride-4 Boundary Residual Refiner
        self.refiner = MultiClassBoundaryResidualRefiner(
            decoder_channels, num_classes
        )

        # 6. Stride-2 High-Resolution Detail Refiner
        self.detail_refiner = MultiClassDetailRefiner(
            decoder_channels, num_classes, detail_channels
        )

    def _global_token(self, global_images: torch.Tensor) -> torch.Tensor:
        global_features = self.rgb_encoder(global_images)
        return global_features[-1].mean(dim=(2, 3))

    def extract_feat(
        self, inputs: torch.Tensor, global_token: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        features = tuple(self.rgb_encoder(inputs))
        if self.use_global_context:
            if global_token is None:
                global_images = F.interpolate(
                    inputs,
                    size=(
                        self.global_thumbnail_size,
                        self.global_thumbnail_size,
                    ),
                    mode='bilinear',
                    align_corners=False,
                )
                global_token = self._global_token(global_images)
            features = self.global_film(features, global_token)
        return features

    def _forward_stages(
        self,
        inputs: torch.Tensor,
        global_token: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        # Multi-scale feature extraction
        features = self.extract_feat(inputs, global_token=global_token)

        # Stride-4 aux logits
        aux_logits = self.aux_head(features[0])

        # Multi-scale decoder
        decoded_features = self.decoder(features)

        # Stride-4 coarse and boundary refinement
        refined = self.refiner(decoded_features)

        # Stride-2 detail refinement
        detail = self.detail_refiner(
            inputs, decoded_features, refined['final']
        )

        boundary_predictions = [detail['boundary'], refined['boundary']]

        return dict(
            final_logits=detail['final'],
            coarse_logits=refined['coarse'],
            aux_logits=aux_logits,
            boundary_predictions=boundary_predictions,
            refinement_gate=detail['refinement_gate'],
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

        pad_t = F.pad(target_f, (1, 1, 1, 1), mode='replicate')
        pad_v = F.pad(valid_f, (1, 1, 1, 1), mode='replicate')

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

    def loss(self, inputs: torch.Tensor, data_samples: Sequence) -> dict:
        global_token = None
        if self.use_global_context:
            global_images = F.interpolate(
                inputs,
                size=(
                    self.global_thumbnail_size,
                    self.global_thumbnail_size,
                ),
                mode='bilinear',
                align_corners=False,
            )
            global_token = self._global_token(global_images)

        outputs = self._forward_stages(inputs, global_token=global_token)

        # Ground truth targets
        targets = torch.stack(
            [sample.gt_sem_seg.data for sample in data_samples], dim=0
        ).to(device=inputs.device)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        targets = targets.long()

        valid = targets != self.ignore_index
        if not valid.any():
            # Fallback guard against completely masked edge patches
            return dict(
                loss_final=outputs['final_logits'].sum() * 0.0,
                loss_coarse=outputs['coarse_logits'].sum() * 0.0,
                loss_aux=outputs['aux_logits'].sum() * 0.0,
                loss_boundary=outputs['boundary_predictions'][0].sum() * 0.0,
            )

        losses = {}

        # 1. Final loss
        final_logits = F.interpolate(
            outputs['final_logits'],
            size=targets.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners,
        )
        losses['loss_final'] = self.loss_weights['final'] * F.cross_entropy(
            final_logits,
            targets,
            ignore_index=self.ignore_index,
            reduction='mean',
        )

        # 2. Coarse loss
        if self.loss_weights.get('coarse', 0.0) > 0.0:
            coarse_logits = F.interpolate(
                outputs['coarse_logits'],
                size=targets.shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )
            losses['loss_coarse'] = self.loss_weights['coarse'] * F.cross_entropy(
                coarse_logits,
                targets,
                ignore_index=self.ignore_index,
                reduction='mean',
            )

        # 3. Aux loss
        if self.loss_weights.get('aux', 0.0) > 0.0:
            aux_logits = F.interpolate(
                outputs['aux_logits'],
                size=targets.shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )
            losses['loss_aux'] = self.loss_weights['aux'] * F.cross_entropy(
                aux_logits,
                targets,
                ignore_index=self.ignore_index,
                reduction='mean',
            )

        # 4. Boundary loss
        if self.loss_weights.get('boundary', 0.0) > 0.0:
            boundary_target, boundary_valid = self._extract_boundary(
                targets, valid
            )
            b_losses = []
            for b_pred in outputs['boundary_predictions']:
                b_target = F.interpolate(
                    boundary_target, size=b_pred.shape[-2:], mode='nearest'
                )
                b_valid = F.interpolate(
                    boundary_valid, size=b_pred.shape[-2:], mode='nearest'
                )
                bce = F.binary_cross_entropy_with_logits(
                    b_pred, b_target, reduction='none'
                )
                b_loss = (bce * b_valid).sum() / b_valid.sum().clamp_min(1.0)
                b_losses.append(b_loss)
            losses['loss_boundary'] = self.loss_weights['boundary'] * (
                sum(b_losses) / len(b_losses)
            )

        return losses

    def encode_decode(
        self,
        inputs: torch.Tensor,
        batch_img_metas: list[dict] | None = None,
        global_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del batch_img_metas
        outputs = self._forward_stages(inputs, global_token=global_token)
        logits = outputs['final_logits']
        return F.interpolate(
            logits,
            size=inputs.shape[-2:],
            mode='bilinear',
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
                inputs,
                size=(
                    self.global_thumbnail_size,
                    self.global_thumbnail_size,
                ),
                mode='bilinear',
                align_corners=False,
            )
            global_token = self._global_token(global_images)
        return self.encode_decode(
            inputs, batch_img_metas, global_token=global_token
        )

    def slide_inference(
        self, inputs: torch.Tensor, batch_img_metas: list[dict]
    ) -> torch.Tensor:
        h_stride, w_stride = self.test_cfg['stride']
        h_crop, w_crop = self.test_cfg['crop_size']
        batch, _, height, width = inputs.shape
        h_grids = max(height - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(width - w_crop + w_stride - 1, 0) // w_stride + 1
        predictions = inputs.new_zeros(batch, self.num_classes, height, width)
        counts = inputs.new_zeros(batch, 1, height, width)
        blend_windows: dict[tuple[int, int], torch.Tensor] = {}

        global_token = None
        if self.use_global_context:
            global_images = F.interpolate(
                inputs,
                size=(
                    self.global_thumbnail_size,
                    self.global_thumbnail_size,
                ),
                mode='bilinear',
                align_corners=False,
            )
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
                    crop, crop_metas, global_token=global_token
                )
                crop_size = crop_logits.shape[-2:]
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
