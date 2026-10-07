"""HDNet MMSeg adapter for binary WWTP and multiclass aerial benchmarks."""
from collections.abc import Sequence

import numpy as np
import torch
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F

from .hd_net import HDNet


def _bce(logits, target, valid, positive_weight=1.0, dice=False):
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
        valid = F.interpolate(valid.float(), size=logits.shape[-2:], mode='nearest').bool()
    logits = logits.float()
    active = valid.float()
    target = torch.where(valid, target, 0)
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


@MODELS.register_module()
class HDNetBaseline(BaseSegmentor):
    def __init__(self, base_channel=48, num_classes=2,
                 data_preprocessor=None, test_cfg=None, init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if num_classes < 2:
            raise ValueError('HDNet needs at least two classes')
        self.num_classes = num_classes
        self.network = HDNet(base_channel, 1 if num_classes == 2 else num_classes)
        self.test_cfg = test_cfg or dict(mode='whole')
        self.align_corners = False

    def extract_feat(self, inputs):
        return self.network(inputs)

    def encode_decode(self, inputs, batch_img_metas=None):
        logits = self.network(inputs)['final']
        logits = F.interpolate(logits, size=inputs.shape[-2:],
                               mode='bilinear', align_corners=False)
        if self.num_classes == 2:
            logits = torch.cat((torch.zeros_like(logits), logits), dim=1)
        return logits

    def _forward(self, inputs, data_samples=None):
        return self.encode_decode(inputs)

    def predict(self, inputs, data_samples=None):
        if self.test_cfg.get('mode', 'whole') == 'whole':
            logits = self.encode_decode(inputs)
        elif self.test_cfg['mode'] == 'slide':
            crop_h, crop_w = self.test_cfg['crop_size']
            stride_h, stride_w = self.test_cfg['stride']
            batch, _, height, width = inputs.shape
            logits = inputs.new_zeros((batch, self.num_classes, height, width))
            coverage = inputs.new_zeros((batch, 1, height, width))
            for y in range(0, max(height - crop_h, 0) + stride_h, stride_h):
                for x in range(0, max(width - crop_w, 0) + stride_w, stride_w):
                    y2, x2 = min(y + crop_h, height), min(x + crop_w, width)
                    y1, x1 = max(y2 - crop_h, 0), max(x2 - crop_w, 0)
                    logits[:, :, y1:y2, x1:x2] += self.encode_decode(inputs[:, :, y1:y2, x1:x2])
                    coverage[:, :, y1:y2, x1:x2] += 1
            logits = logits / coverage.clamp_min(1)
        else:
            raise ValueError(f'Unsupported inference mode: {self.test_cfg["mode"]}')
        return self.postprocess_result(logits, data_samples)

    def _boundary(self, target, valid):
        # Keep the original binary Euclidean-distance supervision for WWTP.
        if self.num_classes == 2:
            masks = target[:, 0].detach().cpu().numpy() > 0.5
            good_masks = valid[:, 0].detach().cpu().numpy()
            boundaries = []
            for mask, good in zip(masks, good_masks):
                mask = mask & good
                if not np.any(mask) or not np.any((~mask) & good):
                    boundaries.append(np.zeros_like(mask, dtype=np.float32))
                else:
                    distance = distance_transform_edt(mask) + distance_transform_edt((~mask) & good)
                    boundaries.append(((distance > 0) & (distance < 3) & good).astype(np.float32))
            return torch.from_numpy(np.stack(boundaries))[:, None].to(target.device)
        # Multiclass boundaries include transitions between any two valid IDs.
        edge = torch.zeros_like(valid)
        vertical = (target[:, :, 1:] != target[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
        horizontal = (target[:, :, :, 1:] != target[:, :, :, :-1]) & valid[:, :, :, 1:] & valid[:, :, :, :-1]
        edge[:, :, 1:] |= vertical
        edge[:, :, :-1] |= vertical
        edge[:, :, :, 1:] |= horizontal
        edge[:, :, :, :-1] |= horizontal
        return F.max_pool2d(edge.float(), 3, stride=1, padding=1) * valid

    def _seg_loss(self, logits, target, valid):
        if self.num_classes == 2:
            return _bce(logits, target.float(), valid, dice=True)
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
        labels = target[:, 0].long()
        # Sum / valid count makes all-ignore batches finite and differentiable.
        loss = F.cross_entropy(logits.float(), labels, ignore_index=255, reduction='none')
        return (loss * valid[:, 0]).sum() / valid.sum().clamp_min(1)

    def loss(self, inputs, data_samples: Sequence):
        outputs = self.network(inputs)
        target = torch.stack([sample.gt_sem_seg.data for sample in data_samples]).to(inputs.device)
        valid = target != 255
        boundary = self._boundary(target, valid)
        losses = dict(
            loss_seg=self._seg_loss(outputs['final'], target, valid),
            loss_boundary=_bce(outputs['boundary'], boundary, valid, 9, dice=True))
        for i, weight in enumerate((0.3, 0.3, 0.5, 0.5, 0.5, 0.5)):
            losses[f'loss_seg_aux{i + 1}'] = weight * self._seg_loss(outputs['segments'][i], target, valid)
            losses[f'loss_boundary_aux{i + 1}'] = weight * _bce(
                outputs['edges'][i], boundary, valid, 9 if i == 5 else 3, dice=True)
        return losses
