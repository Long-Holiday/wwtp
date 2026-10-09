"""Optimized RPGV v6: decoder geometry, normalized bands and residual contours."""

from __future__ import annotations

import math

import torch
from mmseg.registry import MODELS
from torch import nn
from torch.nn import functional as F

from wwtpseg.rpgv_v6_targets import get_contour_targets_v6

from .rpgv_net import _binary_segmentation_loss
from .rpgv_net_v2 import masked_mean, region_and_boundary_loss
from .rpgv_net_v51 import RPGVNetV51
from ..utils.rpgv_v6_decoder import ResidualContourHeadV6, SpatialGeometryDecoderV6
from ..utils.rpgv_v6_modules import (
    DeformableDepthRectifierV6,
    FrequencyGeometryStageV6,
    RegionContrastiveLossV6,
    SmoothFrequencySplitV6,
)


@MODELS.register_module()
class RPGVNetV6(RPGVNetV51):
    """Keep RGB/geometry descriptors independent and fuse into decoder F8/F4.

    Contrastive RGB prototypes read untouched lateral features, not decoder
    features already carrying geometry. Historical v6 checkpoints are not
    compatible with this decoder/projection/contour architecture.
    """

    def __init__(self, geometry_channels=(32, 64), use_geometry=True,
                 geometry_dropout_prob=0.0, rectifier_channels=16,
                 max_offset=2.0, max_depth_correction=0.1,
                 frequency_cutoff=0.15, frequency_padding=8,
                 cluster_cfg=None, attention_cfg=None,
                 fusion_scale=0.1, contrastive_embed_dim=64,
                 contrastive_temperature=0.07, contrastive_loss_weight=0.05,
                 rectification_loss_weight=0.01, fusion_location='decoder',
                 contour_mode='residual', **kwargs):
        if not 0 <= geometry_dropout_prob <= 1:
            raise ValueError('geometry_dropout_prob must be in [0, 1]')
        for name, value in dict(contrastive_loss_weight=contrastive_loss_weight,
                                rectification_loss_weight=rectification_loss_weight,
                                fusion_scale=fusion_scale).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if fusion_location not in {'decoder', 'encoder'}:
            raise ValueError('fusion_location must be decoder or encoder')
        if contour_mode not in {'residual', 'legacy'}:
            raise ValueError('contour_mode must be residual or legacy')
        if kwargs.get('geometry_input', 'depth') != 'depth':
            raise ValueError('v6 requires geometry_input=depth; use_geometry=False is RGB only')
        # Reuse RGB/inference setup, never instantiate the older geometry path.
        super().__init__(geometry_channels=geometry_channels, use_geometry=False,
                         geometry_dropout_prob=0.0, **kwargs)
        self.use_geometry = bool(use_geometry)
        self.geometry_dropout_prob = geometry_dropout_prob if use_geometry else 0.0
        self.contrastive_loss_weight = float(contrastive_loss_weight)
        self.rectification_loss_weight = float(rectification_loss_weight)
        self.fusion_location = fusion_location
        self.contour_mode = contour_mode
        previous = self.decoder
        decoder_channels = previous.projections[0][0].out_channels
        self.decoder = SpatialGeometryDecoderV6(
            self.rgb_channels, decoder_channels, self.geometry_channels,
            use_context=self.use_context, spatial_refinement=self.spatial_refinement)
        self.decoder.load_state_dict(previous.state_dict(), strict=True)
        if contour_mode == 'residual':
            self.refiner = ResidualContourHeadV6(
                decoder_channels, self.refiner.rgb_stem[0][0].out_channels,
                use_contour=self.refiner.use_contour,
                use_sdf=self.loss_weights['sdf'] > 0)
        if self.use_geometry:
            self.depth_rectifier = DeformableDepthRectifierV6(
                rectifier_channels, max_offset, max_depth_correction)
            self.frequency_split = SmoothFrequencySplitV6(frequency_cutoff, frequency_padding)
            source_channels = ((decoder_channels,) * 2 if fusion_location == 'decoder'
                               else self.rgb_channels[:2])
            self.geometry_query_projections = nn.ModuleList([
                nn.Conv2d(r, g, 1, bias=False)
                for r, g in zip(source_channels, self.geometry_channels)
            ])
            self.geometry_stages = nn.ModuleList([
                FrequencyGeometryStageV6(g, cluster_cfg, attention_cfg, fusion_scale)
                for g in self.geometry_channels
            ])
            self.geometry_output_projections = nn.ModuleList([
                nn.Conv2d(g, r, 1, bias=False)
                for r, g in zip(source_channels, self.geometry_channels)
            ])
            if self.contrastive_loss_weight:
                self.geometry_reference_projections = nn.ModuleList([
                    nn.Conv2d(r, g, 1, bias=False)
                    for r, g in zip(source_channels, self.geometry_channels)
                ])
                self.region_contrastive = nn.ModuleList([
                    RegionContrastiveLossV6(g, contrastive_embed_dim, contrastive_temperature)
                    for g in self.geometry_channels
                ])

    def _fuse_geometry(self, index, feature, rgb_reference, low, high, quality):
        query = self.geometry_query_projections[index](feature)
        projection = (self.geometry_reference_projections[index] if self.contrastive_loss_weight
                      else self.geometry_query_projections[index])
        reference = projection(rgb_reference)
        region = self.geometry_stages[index](query, low, high, quality, reference)
        delta = self.geometry_output_projections[index](region['delta'])
        fused = (feature.float() + delta.float()).to(feature.dtype).contiguous()
        return fused, region

    def _predict_from_inputs(self, inputs, geometry_enabled=True):
        rgb, depth, confidence = self._split_inputs(inputs)
        features = tuple(self.rgb_encoder(rgb))
        geometry_outputs = None
        if self.use_geometry and geometry_enabled:
            rectified = self.depth_rectifier(depth, confidence, features[0].shape[-2:])
            quality = rectified['reliability']
            low, high = self.frequency_split(rectified['corrected_depth'], quality)
            regions = [None, None]

            def fuse(index, feature, reference):
                fused, region = self._fuse_geometry(index, feature, reference, low, high, quality)
                regions[index] = region
                return fused

            if self.fusion_location == 'decoder':
                decoded = self.decoder(features, fusion_callback=fuse)
            else:
                # Location ablation only: all other optimized v6 components stay fixed.
                fused = list(features)
                for index in range(2):
                    fused[index] = fuse(index, features[index], features[index])
                decoded = self.decoder(tuple(fused))
            geometry_outputs = dict(rectified=rectified, regions=regions)
        else:
            decoded = self.decoder(features)
        outputs = self.refiner(rgb, decoded)
        outputs.update(seg_logits=self._two_class_logits(outputs['final_logits']),
                       boundary_logits=None, lowres_boundary_logits=None,
                       geometry_v6=geometry_outputs)
        return outputs

    def loss(self, inputs, data_samples):
        target, valid = self._targets(data_samples, inputs.shape[-2:], inputs.device)
        if self.use_geometry and inputs.shape[1] == 5:
            # Unknown/padded labels cannot supply contrastive or rectification targets.
            inputs = torch.cat((inputs[:, :4], inputs[:, 4:5] * valid.to(inputs.dtype)), dim=1)
            inputs = self._apply_geometry_dropout(inputs)
        outputs = self._run_network(inputs)
        logits = F.interpolate(outputs['final_logits'], size=target.shape[-2:],
                               mode='bilinear', align_corners=False)
        losses = dict(loss_final=_binary_segmentation_loss(logits, target, valid))
        losses.update(self._shape_losses(outputs, target, valid, data_samples))
        return losses

    def _shape_losses(self, outputs, target, valid, data_samples=None):
        # V6 owns target caching and disabled-loss elimination; older models keep
        # their original loss APIs and numerical definitions unchanged.
        losses = {}
        coarse = outputs['coarse_logits']
        if self.coarse_loss_weight:
            size = coarse.shape[-2:]
            coarse_valid = F.adaptive_avg_pool2d(valid.float(), size) >= 1.0
            coarse_target = F.adaptive_avg_pool2d(target.float(), size)
            losses['loss_coarse'] = self.coarse_loss_weight * _binary_segmentation_loss(
                coarse, coarse_target, coarse_valid)
        if self.region_loss_weight or self.final_boundary_loss_weight:
            final = F.interpolate(outputs['final_logits'], size=target.shape[-2:],
                                  mode='bilinear', align_corners=False)
            region, boundary = region_and_boundary_loss(final, target, valid)
            if self.region_loss_weight:
                losses['loss_region_consistency'] = self.region_loss_weight * region
            if self.final_boundary_loss_weight:
                losses['loss_final_boundary'] = self.final_boundary_loss_weight * boundary
        if self.loss_weights['sdf']:
            sdf, known = get_contour_targets_v6(
                target, valid, outputs['sdf'].shape[-2:], self.contour_truncation, data_samples)
            error = F.smooth_l1_loss(outputs['sdf'].float().tanh(), sdf, reduction='none')
            losses['loss_sdf'] = self.loss_weights['sdf'] * masked_mean(
                error * (1 + 4 * (1 - sdf.abs())), known)
        if self.boundary_mode == 'band' and self.boundary_band_loss_weight:
            final = F.interpolate(outputs['final_logits'], size=target.shape[-2:],
                                  mode='bilinear', align_corners=False)
            losses['loss_boundary_band'] = self.boundary_band_loss_weight * self._boundary_band_loss(
                final, target, valid)
        geometry = outputs.get('geometry_v6')
        if geometry is None:
            return losses
        if self.contrastive_loss_weight:
            contrastive = [head(region['geometry_prototypes'], region['rgb_prototypes'],
                                region['valid_regions'])
                           for head, region in zip(self.region_contrastive, geometry['regions'])]
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
