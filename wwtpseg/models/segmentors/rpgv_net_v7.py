"""RPGV v7: geometry primitives, relational contrast and structure-aware decoding."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from mmseg.registry import MODELS

from .rpgv_net import _binary_segmentation_loss
from .rpgv_net_v6 import RPGVNetV6
from ..utils.rpgv_v6_modules import _resize
from ..utils.rpgv_v7_decoder import SpatialGeometryDecoderV7
from ..utils.rpgv_v7_losses import HierarchicalContrastV7
from ..utils.rpgv_v7_modules import GeometryPrimitivesV7, GeometryRelationNetworkV7


@MODELS.register_module()
class RPGVNetV7(RPGVNetV6):
    def __init__(self, structure_channels=128, primitive_cfg=None, graph_cfg=None,
                 hierarchical_cfg=None, structure_feedback=True,
                 structure_aux_weight=0.1, structure_source='depth', **kwargs):
        if structure_source not in {'depth', 'rgb'}:
            raise ValueError('structure_source must be depth or rgb')
        if not math.isfinite(structure_aux_weight) or structure_aux_weight < 0:
            raise ValueError('structure auxiliary weight must be nonnegative')
        if kwargs.get('fusion_location', 'decoder') != 'decoder':
            raise ValueError('v7 requires decoder geometry fusion')
        if kwargs.get('contour_mode', 'residual') != 'residual':
            raise ValueError('v7 requires residual contours')
        # The hierarchical node objective replaces, rather than doubles, v6 alignment.
        kwargs['contrastive_loss_weight'] = 0.0
        super().__init__(**kwargs)
        self.structure_feedback = bool(structure_feedback)
        self.structure_aux_weight = float(structure_aux_weight)
        self.structure_source = structure_source
        channels = self.decoder.projections[0][0].out_channels
        previous = self.decoder
        self.decoder = SpatialGeometryDecoderV7(
            self.rgb_channels, channels, self.geometry_channels,
            use_context=self.use_context, spatial_refinement=self.spatial_refinement)
        self.decoder.load_state_dict(previous.state_dict(), strict=True)
        if self.use_geometry:
            self.primitives = GeometryPrimitivesV7(structure_channels, **(primitive_cfg or {}))
            self.relation_graph = GeometryRelationNetworkV7(structure_channels, **(graph_cfg or {}))
            self.hierarchical = HierarchicalContrastV7(structure_channels, **(hierarchical_cfg or {}))
            if self.hierarchical.enabled:
                self.structure_rgb_projections = nn.ModuleList([
                    nn.Conv2d(c, structure_channels, 1) for c in self.rgb_channels[1:]])
            if self.structure_feedback:
                self.structure_fusion = nn.ModuleList([
                    nn.Conv2d(structure_channels + channels, channels, 1) for _ in range(2)])
            if self.structure_aux_weight:
                self.structure_head = nn.Conv2d(structure_channels, self.refiner.num_classes, 1)

    def _structure_input(self, inputs):
        if self.structure_source != 'rgb':
            return inputs
        # Capacity control ignores BOTH supplied geometry channels, including missing Q.
        gray = inputs[:, :3].float().mean(1, keepdim=True).clamp(0, 255)
        return torch.cat((inputs[:, :3], gray.to(inputs.dtype), torch.full_like(gray, 255)), 1)

    def _predict_from_inputs(self, inputs, geometry_enabled=True, prepared=False):
        if not prepared:
            inputs = self._structure_input(inputs)
        rgb, depth, confidence = self._split_inputs(inputs)
        features = tuple(self.rgb_encoder(rgb))
        geometry, structure = None, None
        if self.use_geometry and geometry_enabled:
            rectified = self.depth_rectifier(depth, confidence, features[0].shape[-2:])
            quality = rectified['reliability']
            low, high = self.frequency_split(rectified['corrected_depth'], quality)
            primitives = self.primitives(rectified['corrected_depth'], low, high, quality,
                                         measured_depth=rectified['base_depth'].detach())
            nodes, adjacency = self.relation_graph(
                primitives['nodes'], primitives['content'], primitives['positions'],
                primitives['contact'], primitives['valid'])
            with torch.autocast(device_type=nodes.device.type, enabled=False):
                dense = torch.bmm(nodes.transpose(1, 2), primitives['assignment'].flatten(2))
                dense = dense.reshape(nodes.shape[0], nodes.shape[-1], *primitives['assignment'].shape[-2:])
                dense = dense * primitives['support']
            regions = [None, None]

            def fuse(index, feature, reference):
                fused, region = self._fuse_geometry(index, feature, reference, low, high, quality)
                regions[index] = region
                return fused

            def fuse_structure(index, feature):
                observed = _resize(primitives['support'], feature.shape[-2:]) > 0
                context = _resize(dense, feature.shape[-2:]).to(feature.dtype)
                update = self.structure_fusion[index](torch.cat((feature, context), 1))
                return (feature.float() + update.float() * observed).to(feature.dtype).contiguous()

            decoded = self.decoder(features, fusion_callback=fuse,
                                   structure_callback=fuse_structure if self.structure_feedback else None)
            geometry = dict(rectified=rectified, regions=regions)
            structure = dict(primitives=primitives, adjacency=adjacency, dense=dense,
                             rgb_features=features[1:])
            if self.structure_aux_weight:
                structure['logits'] = self.structure_head(dense)
        else:
            decoded = self.decoder(features)
        outputs = self.refiner(rgb, decoded)
        outputs.update(seg_logits=self._segmentation_logits(outputs['final_logits']),
                       boundary_logits=None, lowres_boundary_logits=None,
                       geometry_v6=geometry, structure_v7=structure)
        return outputs

    def loss(self, inputs, data_samples):
        target, valid = self._targets(data_samples, inputs.shape[-2:], inputs.device)
        inputs = self._structure_input(inputs)
        if self.use_geometry and inputs.shape[1] == 5:
            inputs = torch.cat((inputs[:, :4], inputs[:, 4:5] * valid.to(inputs.dtype)), 1)
            inputs = self._apply_geometry_dropout(inputs)
        outputs = self._predict_from_inputs(inputs, prepared=True)
        logits = F.interpolate(outputs['final_logits'], size=target.shape[-2:],
                               mode='bilinear', align_corners=False)
        loss = (_binary_segmentation_loss(logits, target, valid) if self.refiner.num_classes == 1
                else self._cross_entropy(logits, target, valid))
        losses = dict(loss_final=loss)
        losses.update(self._shape_losses(outputs, target, valid, data_samples))
        return losses

    def _shape_losses(self, outputs, target, valid, data_samples=None):
        # Cooperative inheritance uses the existing multiclass shape objectives in its adapter.
        losses = super()._shape_losses(outputs, target, valid, data_samples)
        structure = outputs['structure_v7']
        if structure is None:
            return losses
        primitives = structure['primitives']
        if self.hierarchical.enabled:
            size = primitives['assignment'].shape[-2:]
            appearance = sum(_resize(projection(feature), size) for projection, feature in zip(
                self.structure_rgb_projections, structure['rgb_features'])) / 3
            rgb_nodes = self.primitives.pool(appearance, primitives['pooling'])
            losses.update(self.hierarchical(primitives, rgb_nodes, self.relation_graph,
                                            structure['adjacency']))
        if self.structure_aux_weight:
            logits = structure['logits']
            size = logits.shape[-2:]
            observed = (F.adaptive_avg_pool2d(valid.float(), size) >= 1) & (primitives['support'] > 0)
            if self.refiner.num_classes == 1:
                truth = F.adaptive_avg_pool2d(target.float(), size)
                auxiliary = _binary_segmentation_loss(logits, truth, observed)
            else:
                truth = F.interpolate(target.float(), size=size, mode='nearest').long()
                auxiliary = self._cross_entropy(logits, truth, observed)
            losses['loss_structure_aux_v7'] = self.structure_aux_weight * auxiliary
        return losses
