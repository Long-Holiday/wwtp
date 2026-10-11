from __future__ import annotations

import torch
from mmseg.registry import MODELS
from torch.nn import functional as F

from wwtpseg.rpgv_v6_targets import contour_target_v6

from ..rpgv_net_v6 import RPGVNetV6
from ..rpgv_net_v2 import masked_mean
from ...utils.rpgv_v6_decoder import ResidualContourHeadV6
from .model import RPGVRemoteSensing


@MODELS.register_module()
class RPGVRemoteSensingV6(RPGVNetV6):
    def __init__(self, num_classes, ignore_index=255, **kwargs):
        if num_classes < 2:
            raise ValueError('num_classes must be at least 2')
        if kwargs.get('contour_mode', 'residual') != 'residual':
            raise ValueError('multiclass v6 requires contour_mode=residual')
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        previous = self.refiner
        self.refiner = ResidualContourHeadV6(
            previous.coarse_head.in_channels,
            previous.rgb_stem[0][0].out_channels if hasattr(previous, 'rgb_stem')
            else kwargs.get('detail_channels', 24),
            use_contour=previous.use_contour, use_sdf=previous.use_sdf,
            num_classes=num_classes)

    def _segmentation_logits(self, logits):
        return logits

    def _targets(self, data_samples, size, device):
        labels = torch.stack([sample.gt_sem_seg.data for sample in data_samples]).to(device)
        labels = F.interpolate(labels.float(), size=size, mode='nearest').long()
        valid = labels != self.ignore_index
        return labels.masked_fill(~valid, 0), valid

    @staticmethod
    def _cross_entropy(logits, target, valid):
        error = F.cross_entropy(logits.float(), target[:, 0].long(), reduction='none')
        return masked_mean(error, valid[:, 0])

    def loss(self, inputs, data_samples):
        target, valid = self._targets(data_samples, inputs.shape[-2:], inputs.device)
        if self.use_geometry and inputs.shape[1] == 5:
            inputs = torch.cat((inputs[:, :4], inputs[:, 4:5] * valid.to(inputs.dtype)), dim=1)
            inputs = self._apply_geometry_dropout(inputs)
        outputs = self._run_network(inputs)
        final = F.interpolate(outputs['final_logits'], size=target.shape[-2:],
                              mode='bilinear', align_corners=False)
        losses = dict(loss_final=self._cross_entropy(final, target, valid))
        losses.update(self._shape_losses(outputs, target, valid))
        return losses

    def _shape_losses(self, outputs, target, valid, data_samples=None):
        losses = {}
        coarse = outputs['coarse_logits']
        if self.coarse_loss_weight:
            size = coarse.shape[-2:]
            coarse_valid = F.adaptive_avg_pool2d(valid.float(), size) >= 1.0
            coarse_target = F.interpolate(target.float(), size=size, mode='nearest').long()
            losses['loss_coarse'] = self.coarse_loss_weight * self._cross_entropy(
                coarse, coarse_target, coarse_valid)
        final = F.interpolate(outputs['final_logits'], size=target.shape[-2:],
                              mode='bilinear', align_corners=False)
        if self.region_loss_weight or self.final_boundary_loss_weight:
            probability = final.float().softmax(dim=1)
            one_hot = F.one_hot(target[:, 0], self.num_classes).permute(0, 3, 1, 2).float()
            region, boundary = probability.sum() * 0, probability.sum() * 0
            for dimension in (-2, -1):
                left, right = [slice(None)] * 4, [slice(None)] * 4
                left[dimension], right[dimension] = slice(None, -1), slice(1, None)
                a, b = tuple(left), tuple(right)
                pairs = valid[a] & valid[b]
                same = pairs & (target[a] == target[b])
                different = pairs & (target[a] != target[b])
                delta = probability[b] - probability[a]
                truth = one_hot[b] - one_hot[a]
                region = region + masked_mean(delta.abs().mean(1, keepdim=True), same)
                boundary = boundary + masked_mean((delta - truth).abs().mean(1, keepdim=True), different)
            if self.region_loss_weight:
                losses['loss_region_consistency'] = self.region_loss_weight * region / 2
            if self.final_boundary_loss_weight:
                losses['loss_final_boundary'] = self.final_boundary_loss_weight * boundary / 2
        if self.loss_weights['sdf']:
            terms = []
            for index in range(self.num_classes):
                sdf, known = contour_target_v6(
                    (target == index).float() * valid, valid,
                    outputs['sdf'].shape[-2:], self.contour_truncation)
                error = F.smooth_l1_loss(
                    outputs['sdf'][:, index:index + 1].float().tanh(), sdf, reduction='none')
                terms.append(masked_mean(error * (1 + 4 * (1 - sdf.abs())), known))
            losses['loss_sdf'] = self.loss_weights['sdf'] * torch.stack(terms).mean()
        if self.boundary_mode == 'band' and self.boundary_band_loss_weight:
            losses['loss_boundary_band'] = self.boundary_band_loss_weight * self._boundary_band_loss(
                final, target, valid)
        losses.update(self._v6_geometry_losses(outputs))
        return losses

    def _boundary_band_loss(self, logits, target, valid):
        edge = torch.zeros_like(valid)
        horizontal = (target[..., :, 1:] != target[..., :, :-1]) & (
            valid[..., :, 1:] & valid[..., :, :-1])
        vertical = (target[..., 1:, :] != target[..., :-1, :]) & (
            valid[..., 1:, :] & valid[..., :-1, :])
        edge[..., :, 1:] |= horizontal
        edge[..., :, :-1] |= horizontal
        edge[..., 1:, :] |= vertical
        edge[..., :-1, :] |= vertical
        radius = self.boundary_band_radius
        kernel = 2 * radius + 1
        band = F.max_pool2d(edge.float(), kernel, 1, radius) > 0
        near_ignore = F.max_pool2d((~valid).float(), kernel, 1, radius) > 0
        band = band & valid & ~near_ignore
        error = F.cross_entropy(logits.float(), target[:, 0], reduction='none')
        terms, present = [], []
        for index in range(self.num_classes):
            selected = band[:, 0] & (target[:, 0] == index)
            terms.append(masked_mean(error, selected))
            present.append(selected.any())
        return torch.stack(terms).sum() / torch.stack(present).sum().clamp_min(1)

    def slide_inference(self, inputs, batch_img_metas):
        return RPGVRemoteSensing.slide_inference(self, inputs, batch_img_metas)
