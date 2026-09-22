"""MMSeg adapters and mask-derived supervision for the edge baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from scipy.ndimage import distance_transform_edt
from torch import nn
from torch.nn import functional as F

from .cbr_net import CBRNet
from .hd_net import HDNet


def _target(samples: Sequence, device: torch.device
            ) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.stack([sample.gt_sem_seg.data for sample in samples], 0)
    labels = labels.to(device=device)
    valid = labels != 255
    return torch.where(valid, labels, 0).float(), valid


def _boundary_and_direction(labels: torch.Tensor, valid: torch.Tensor,
                            radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute supervision after augmentation, without writing dataset files."""
    masks = labels[:, 0].detach().cpu().numpy() > 0.5
    valid_masks = valid[:, 0].detach().cpu().numpy()
    boundaries = []
    directions = []
    offsets = np.asarray(CBRNet.DIRECTIONS, dtype=np.float32)
    offsets /= np.linalg.norm(offsets, axis=1, keepdims=True)
    for mask, good in zip(masks, valid_masks):
        if not np.any(mask & good) or not np.any((~mask) & good):
            boundaries.append(np.zeros_like(mask, dtype=np.float32))
            directions.append(np.full_like(mask, -1, dtype=np.int64))
            continue
        mask = mask & good
        distance = distance_transform_edt(mask) + \
            distance_transform_edt((~mask) & good)
        edge = (distance > 0) & (distance < radius) & good
        dy, dx = np.gradient(distance.astype(np.float32))
        angle_scores = dy[..., None] * offsets[:, 0] + \
            dx[..., None] * offsets[:, 1]
        classes = angle_scores.argmax(axis=-1).astype(np.int64)
        classes[~edge] = -1
        boundaries.append(edge.astype(np.float32))
        directions.append(classes)
    boundary = torch.from_numpy(np.stack(boundaries))[:, None].to(labels.device)
    direction = torch.from_numpy(np.stack(directions)).to(labels.device)
    return boundary, direction


def _bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
         positive_weight: float = 1.0, dice: bool = False) -> torch.Tensor:
    size = logits.shape[-2:]
    if target.shape[-2:] != size:
        target = F.interpolate(target, size=size, mode='nearest')
        valid = F.interpolate(valid.float(), size=size, mode='nearest').bool()
    logits = logits.float()
    active = valid.float()
    element = F.binary_cross_entropy_with_logits(
        logits, target.float(), reduction='none',
        pos_weight=logits.new_tensor(positive_weight))
    loss = (element * active).sum() / active.sum().clamp_min(1)
    if dice:
        probability = logits.sigmoid() * active
        target = target.float() * active
        intersection = (probability * target).sum(dim=(1, 2, 3))
        totals = (probability + target).sum(dim=(1, 2, 3))
        loss = loss + (1 - (2 * intersection + 1) / (totals + 1)).mean()
    return loss


class _EdgeSegmentor(BaseSegmentor):
    def __init__(self, network: nn.Module, edge_radius: int,
                 data_preprocessor: dict | nn.Module | None = None,
                 test_cfg: dict | None = None,
                 init_cfg: dict | None = None) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.network = network
        self.edge_radius = edge_radius
        self.test_cfg = test_cfg or dict(mode='whole')
        self.align_corners = False

    def extract_feat(self, inputs: torch.Tensor) -> dict:
        return self.network(inputs)

    def _forward(self, inputs: torch.Tensor,
                 data_samples: Sequence | None = None) -> torch.Tensor:
        return self.encode_decode(inputs)

    def encode_decode(self, inputs: torch.Tensor,
                      batch_img_metas: Sequence | None = None) -> torch.Tensor:
        del batch_img_metas
        foreground = self.network(inputs)['final']
        foreground = F.interpolate(foreground, size=inputs.shape[-2:],
                                   mode='bilinear', align_corners=False)
        return torch.cat((torch.zeros_like(foreground), foreground), dim=1)

    def predict(self, inputs: torch.Tensor,
                data_samples: Sequence | None = None) -> list:
        if self.test_cfg.get('mode', 'whole') == 'whole':
            logits = self.encode_decode(inputs)
        elif self.test_cfg['mode'] == 'slide':
            crop_h, crop_w = self.test_cfg['crop_size']
            stride_h, stride_w = self.test_cfg['stride']
            batch, _, height, width = inputs.shape
            h_count = max(height - crop_h + stride_h - 1, 0) // stride_h + 1
            w_count = max(width - crop_w + stride_w - 1, 0) // stride_w + 1
            logits = inputs.new_zeros((batch, 2, height, width))
            coverage = inputs.new_zeros((batch, 1, height, width))
            for row in range(h_count):
                for col in range(w_count):
                    y2 = min(row * stride_h + crop_h, height)
                    x2 = min(col * stride_w + crop_w, width)
                    y1 = max(y2 - crop_h, 0)
                    x1 = max(x2 - crop_w, 0)
                    crop = inputs[:, :, y1:y2, x1:x2]
                    logits[:, :, y1:y2, x1:x2] += self.encode_decode(crop)
                    coverage[:, :, y1:y2, x1:x2] += 1
            logits /= coverage.clamp_min(1)
        else:
            raise ValueError(f'unsupported inference mode: {self.test_cfg["mode"]}')
        return self.postprocess_result(logits, data_samples)


@MODELS.register_module()
class CBRNetBaseline(_EdgeSegmentor):
    def __init__(self, pretrained: bool = True,
                 data_preprocessor: dict | nn.Module | None = None,
                 test_cfg: dict | None = None,
                 init_cfg: dict | None = None) -> None:
        super().__init__(CBRNet(pretrained=pretrained), edge_radius=5,
                         data_preprocessor=data_preprocessor,
                         test_cfg=test_cfg, init_cfg=init_cfg)

    def loss(self, inputs: torch.Tensor, data_samples: Sequence) -> dict:
        outputs = self.network(inputs)
        target, valid = _target(data_samples, inputs.device)
        edge, direction = _boundary_and_direction(
            target, valid, self.edge_radius)
        segments = outputs['segments']
        edges = outputs['edges']
        losses = dict(
            loss_refined=_bce(outputs['final'], target, valid),
            loss_seg=_bce(segments[-1], target, valid),
            loss_pseudo=_bce(segments[-1],
                             outputs['final'].detach().sigmoid(), valid),
            loss_edge=_bce(edges[-1], edge, valid, positive_weight=0.85),
        )
        for index, prediction in enumerate(segments[:-1]):
            losses[f'loss_seg_aux{index + 1}'] = 0.25 * _bce(
                prediction, target, valid)
        for index, prediction in enumerate(edges[:-1]):
            losses[f'loss_edge_aux{index + 1}'] = 0.25 * _bce(
                prediction, edge, valid, positive_weight=0.85)
        direction_valid = direction != -1
        if direction_valid.any():
            direction_loss = F.cross_entropy(
                outputs['direction'].float(), direction,
                ignore_index=-1, reduction='none')
            losses['loss_direction'] = (direction_loss * direction_valid).sum() / \
                direction_valid.sum()
        else:
            losses['loss_direction'] = outputs['direction'].sum() * 0.0
        return losses


@MODELS.register_module()
class HDNetBaseline(_EdgeSegmentor):
    def __init__(self, base_channel: int = 48,
                 data_preprocessor: dict | nn.Module | None = None,
                 test_cfg: dict | None = None,
                 init_cfg: dict | None = None) -> None:
        super().__init__(HDNet(base_channel=base_channel), edge_radius=3,
                         data_preprocessor=data_preprocessor,
                         test_cfg=test_cfg, init_cfg=init_cfg)

    def loss(self, inputs: torch.Tensor, data_samples: Sequence) -> dict:
        outputs = self.network(inputs)
        target, valid = _target(data_samples, inputs.device)
        boundary, _ = _boundary_and_direction(
            target, valid, self.edge_radius)
        losses = dict(
            loss_seg=_bce(outputs['final'], target, valid, dice=True),
            loss_boundary=_bce(outputs['boundary'], boundary, valid,
                               positive_weight=9, dice=True),
        )
        weights = (0.3, 0.3, 0.5, 0.5, 0.5, 0.5)
        for index, (prediction, weight) in enumerate(
                zip(outputs['segments'], weights)):
            losses[f'loss_seg_aux{index + 1}'] = weight * _bce(
                prediction, target, valid, dice=True)
        for index, (prediction, weight) in enumerate(
                zip(outputs['edges'], weights)):
            positive_weight = 9 if index == 5 else 3
            losses[f'loss_boundary_aux{index + 1}'] = weight * _bce(
                prediction, boundary, valid, positive_weight,
                dice=True)
        return losses
