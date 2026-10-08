"""RPGV v6: deformable depth, frequency geometry, and regional contrastive alignment."""

from __future__ import annotations

import math

import torch
from mmseg.registry import MODELS
from torch import nn

from .rpgv_net_v51 import RPGVNetV51
from ..utils.rpgv_v6_modules import (
    DeformableDepthRectifierV6,
    FrequencyGeometryStageV6,
    RegionContrastiveLossV6,
    SmoothFrequencySplitV6,
)


@MODELS.register_module()
class RPGVNetV6(RPGVNetV51):
    """Keep the RGB encoder independent and fuse geometry into its finest outputs.

    Geometry prototypes never read RGB. Matched RGB prototypes use exactly the
    same spatial assignments, with per-rank in-batch negatives during training.
    The v5.1 spatial decoder, contour supervision and inference APIs are reused.
    """

    def __init__(self, geometry_channels=(32, 64), use_geometry=True,
                 geometry_dropout_prob=0.1, rectifier_channels=16,
                 max_offset=2.0, max_depth_correction=0.1,
                 frequency_cutoff=0.15, frequency_padding=8,
                 cluster_cfg=None, attention_cfg=None,
                 fusion_scale=0.1, contrastive_embed_dim=64,
                 contrastive_temperature=0.07, contrastive_loss_weight=0.05,
                 rectification_loss_weight=0.01, **kwargs):
        if not 0 <= geometry_dropout_prob <= 1:
            raise ValueError('geometry_dropout_prob must be in [0, 1]')
        for name, value in dict(contrastive_loss_weight=contrastive_loss_weight,
                                rectification_loss_weight=rectification_loss_weight,
                                fusion_scale=fusion_scale).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if kwargs.get('geometry_input', 'depth') != 'depth':
            raise ValueError('v6 requires geometry_input=depth; use_geometry=False is RGB only')
        # Geometry fusion is owned by v6; do not construct dormant v5 geometry modules.
        super().__init__(geometry_channels=geometry_channels, use_geometry=False,
                         geometry_dropout_prob=0.0, **kwargs)
        self.use_geometry = bool(use_geometry)
        self.geometry_dropout_prob = geometry_dropout_prob if use_geometry else 0.0
        self.contrastive_loss_weight = float(contrastive_loss_weight)
        self.rectification_loss_weight = float(rectification_loss_weight)
        if self.use_geometry:
            self.depth_rectifier = DeformableDepthRectifierV6(
                rectifier_channels, max_offset, max_depth_correction)
            self.frequency_split = SmoothFrequencySplitV6(frequency_cutoff, frequency_padding)
            self.geometry_rgb_projections = nn.ModuleList([
                nn.Conv2d(r, g, 1, bias=False)
                for r, g in zip(self.rgb_channels[:2], self.geometry_channels)
            ])
            self.geometry_stages = nn.ModuleList([
                FrequencyGeometryStageV6(g, cluster_cfg, attention_cfg, fusion_scale)
                for g in self.geometry_channels
            ])
            self.geometry_output_projections = nn.ModuleList([
                nn.Conv2d(g, r, 1, bias=False)
                for r, g in zip(self.rgb_channels[:2], self.geometry_channels)
            ])
            if self.contrastive_loss_weight:
                self.region_contrastive = nn.ModuleList([
                    RegionContrastiveLossV6(g, contrastive_embed_dim, contrastive_temperature)
                    for g in self.geometry_channels
                ])

    def _predict_from_inputs(self, inputs, geometry_enabled=True):
        rgb, depth, confidence = self._split_inputs(inputs)
        features = list(self.rgb_encoder(rgb))
        geometry_outputs = None
        if self.use_geometry and geometry_enabled:
            rectified = self.depth_rectifier(depth, confidence, features[0].shape[-2:])
            low, high = self.frequency_split(rectified['corrected_depth'])
            regions = []
            for i, (projection, stage, output_projection) in enumerate(zip(
                    self.geometry_rgb_projections, self.geometry_stages,
                    self.geometry_output_projections)):
                rgb_feature = projection(features[i])
                region = stage(rgb_feature, low, high, rectified['reliability'])
                features[i] = features[i] + output_projection(region['fused'] - rgb_feature)
                regions.append(region)
            geometry_outputs = dict(rectified=rectified, regions=regions)
        decoded = self.decoder(tuple(features))
        outputs = self.refiner(rgb, decoded)
        outputs.update(
            seg_logits=self._two_class_logits(outputs['final_logits']),
            boundary_logits=None,
            lowres_boundary_logits=None,
            geometry_v6=geometry_outputs,
        )
        return outputs

    def loss(self, inputs, data_samples):
        if self.use_geometry and inputs.shape[1] == 5:
            _, valid = self._targets(data_samples, inputs.shape[-2:], inputs.device)
            # Padding/ignore pixels cannot create contrastive positives or negatives.
            inputs = torch.cat((inputs[:, :4], inputs[:, 4:5] * valid.to(inputs.dtype)), dim=1)
        return super().loss(inputs, data_samples)

    def _shape_losses(self, outputs, target, valid):
        losses = super()._shape_losses(outputs, target, valid)
        geometry = outputs.get('geometry_v6')
        if geometry is None:
            return losses
        if self.contrastive_loss_weight:
            contrastive = []
            for head, regions in zip(self.region_contrastive, geometry['regions']):
                contrastive.append(head(regions['geometry_prototypes'],
                                        regions['rgb_prototypes'], regions['valid_regions']))
            losses['loss_region_contrastive'] = (
                self.contrastive_loss_weight * torch.stack(contrastive).mean())
            losses['region_valid_count'] = torch.stack([
                r['valid_regions'].float().sum() for r in geometry['regions']
            ]).mean().detach()
        if self.rectification_loss_weight:
            rectified = geometry['rectified']
            quality = rectified['reliability'].float()
            correction = rectified['correction'].float() / self.depth_rectifier.max_correction
            offset = rectified['offsets'].float() / self.depth_rectifier.max_offset
            penalty = correction.square() + offset.square().mean(1, keepdim=True)
            losses['loss_depth_rectification'] = (
                self.rectification_loss_weight * (penalty * quality).sum()
                / quality.sum().clamp_min(1.0))
        return losses
