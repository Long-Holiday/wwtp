"""RGB-only RPGV v8.1 with content-adaptive supports and explicit relation messages."""
from __future__ import annotations

import math

import torch
from mmseg.registry import MODELS
from torch import nn
from torch.nn import functional as F

from .rpgv_net_v8 import RPGVNetV8
from ..utils.rpgv_v8_losses import (boundary_band_loss, gaussian_semantic_losses,
                                   semantic_loss, sdf_loss)
from ..utils.rpgv_v8_modules import GaussianSplatV8
from ..utils.rpgv_v81_losses import (balanced_boundary_band_loss, final_pair_losses,
                                    gaussian_dense_dice_loss)
from ..utils.rpgv_v81_modules import (GaussianPrimitivePredictorV81, GaussianRelationNetworkV81,
                                     LocalGaussianFusionV81)


@MODELS.register_module()
class RPGVNetV81(RPGVNetV8):
    def __init__(self, rgb_encoder, rgb_channels=(64, 128, 320, 512), decoder_channels=64,
                 gaussian_channels=128, gaussian_cfg=None, graph_cfg=None, local_detail=True,
                 region_loss_weight=0.05, final_boundary_loss_weight=0.1,
                 balanced_boundary=True, gaussian_dense_dice_weight=0.05, **kwargs):
        for value in (region_loss_weight, final_boundary_loss_weight, gaussian_dense_dice_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError('v8.1 loss weights must be finite and nonnegative')
        gaussian_cfg, graph_cfg = dict(gaussian_cfg or {}), dict(graph_cfg or {})
        base_gaussian = {key: value for key, value in gaussian_cfg.items()
                         if key not in ('content_aware', 'content_scale')}
        base_graph = {key: value for key, value in graph_cfg.items() if key != 'use_edge_value'}
        super().__init__(rgb_encoder=rgb_encoder, rgb_channels=rgb_channels,
                         decoder_channels=decoder_channels, gaussian_channels=gaussian_channels,
                         gaussian_cfg=base_gaussian, graph_cfg=base_graph, **kwargs)
        self.local_detail = bool(local_detail)
        self.region_loss_weight = float(region_loss_weight)
        self.final_boundary_loss_weight = float(final_boundary_loss_weight)
        self.balanced_boundary = bool(balanced_boundary)
        self.gaussian_dense_dice_weight = float(gaussian_dense_dice_weight) if self.binary else 0.0
        if self.use_gaussians:
            self.gaussian_predictor = GaussianPrimitivePredictorV81(
                rgb_channels, channels=gaussian_channels, **gaussian_cfg)
            if self.use_message_passing:
                self.gaussian_graph = GaussianRelationNetworkV81(
                    channels=gaussian_channels, **graph_cfg)
            if self.local_detail:
                self.local_fusion = nn.ModuleList([
                    LocalGaussianFusionV81(rgb_channels[index], decoder_channels, gaussian_channels + 3)
                    for index in range(2)])
            if self.gaussian_dense_dice_weight and not self.gaussian_dense_weight:
                self.gaussian_dense_head = nn.Conv2d(gaussian_channels + 3, 1, 1)

    def _run_network(self, inputs):
        rgb = self._normalize_bgr(inputs)
        features = self.rgb_encoder(rgb)
        gaussian = None
        if self.use_gaussians:
            primitives, raster, nodes = self.gaussian_predictor(features)
            attributes = torch.cat((raster['height'], raster['coverage'], raster['variance']), dim=1)
            local = torch.cat((GaussianSplatV8.render(nodes, raster), attributes), dim=1)
            graph_nodes = self.gaussian_graph(nodes, primitives, raster) if self.use_message_passing else nodes
            structure = GaussianSplatV8.render(graph_nodes, raster)

            def fuse_local(index, value, lateral):
                if self.local_detail:
                    return self.local_fusion[index](value, local, features[index])
                return self.local_fusion[index](value, local)

            decoded = self.decoder(
                features, fusion_callback=fuse_local,
                structure_callback=lambda index, value: self.structure_fusion[index](value, structure))
            gaussian = dict(primitives=primitives, raster=raster, nodes=graph_nodes)
        else:
            decoded = self.decoder(features)
        outputs = self.refiner(rgb, decoded)
        outputs['gaussian'] = gaussian
        if gaussian is not None:
            if self.gaussian_dense_weight or self.gaussian_dense_dice_weight:
                outputs['gaussian_dense_logits'] = self.gaussian_dense_head(torch.cat((structure, attributes), dim=1))
            if self.gaussian_node_weight:
                outputs['gaussian_node_logits'] = self.gaussian_node_head(graph_nodes)
        logits = outputs['final_logits']
        outputs['seg_logits'] = torch.cat((-0.5 * logits, 0.5 * logits), dim=1) if self.binary else logits
        return outputs

    def loss(self, inputs, data_samples):
        labels = torch.stack([sample.gt_sem_seg.data for sample in data_samples]).to(inputs.device).long()
        if labels.shape[-2:] != inputs.shape[-2:]:
            raise ValueError('v8.1 expects targets padded to the input image shape')
        valid = labels != self.ignore_index
        target = labels.masked_fill(~valid, 0)
        outputs = self._run_network(inputs)
        final = F.interpolate(outputs['final_logits'], size=target.shape[-2:], mode='bilinear', align_corners=False)
        losses = dict(loss_final=semantic_loss(final, target, valid, self.binary))
        if self.coarse_loss_weight:
            coarse = outputs['coarse_logits']
            known = F.adaptive_avg_pool2d(valid.float(), coarse.shape[-2:]) >= 1.0
            if self.binary:
                coarse_target = F.adaptive_avg_pool2d(target.float(), coarse.shape[-2:])
            else:
                coarse_target = F.interpolate(target.float(), size=coarse.shape[-2:], mode='nearest').long()
            losses['loss_coarse'] = self.coarse_loss_weight * semantic_loss(coarse, coarse_target, known, self.binary)
        if self.region_loss_weight or self.final_boundary_loss_weight:
            region, boundary = final_pair_losses(final, target, valid, self.binary)
            if self.region_loss_weight:
                losses['loss_region_consistency'] = self.region_loss_weight * region
            if self.final_boundary_loss_weight:
                losses['loss_final_boundary'] = self.final_boundary_loss_weight * boundary
        if self.sdf_loss_weight:
            losses['loss_sdf'] = self.sdf_loss_weight * sdf_loss(
                outputs['sdf'], target, valid, self.binary, self.contour_truncation)
        if self.boundary_band_loss_weight:
            objective = balanced_boundary_band_loss if self.balanced_boundary else boundary_band_loss
            losses['loss_boundary_band'] = self.boundary_band_loss_weight * objective(
                final, target, valid, self.binary, self.boundary_band_radius)
        if self.use_gaussians:
            if self.gaussian_dense_weight or self.gaussian_node_weight:
                losses.update(gaussian_semantic_losses(outputs, target, valid, self.binary,
                                                       self.gaussian_dense_weight, self.gaussian_node_weight))
            if self.gaussian_dense_dice_weight:
                losses['loss_gaussian_dense_dice'] = self.gaussian_dense_dice_weight * gaussian_dense_dice_loss(
                    outputs['gaussian_dense_logits'], target, valid)
        return losses
