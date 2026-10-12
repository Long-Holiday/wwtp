"""RPGV v8: RGB-only Gaussian geometry learned jointly with segmentation."""
from __future__ import annotations

import copy

import torch
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from torch import nn
from torch.nn import functional as F

from ..utils.rpgv_v6_decoder import ResidualContourHeadV6
from ..utils.rpgv_v7_decoder import SpatialGeometryDecoderV7
from ..utils.rpgv_v8_losses import (boundary_band_loss, gaussian_semantic_losses,
                                   semantic_loss, sdf_loss)
from ..utils.rpgv_v8_modules import (GaussianFusionV8, GaussianPrimitivePredictorV8,
                                    GaussianRelationNetworkV8, GaussianSplatV8)


@MODELS.register_module()
class RPGVNetV8(BaseSegmentor):
    def __init__(self, rgb_encoder, rgb_channels=(64, 128, 320, 512),
                 decoder_channels=64, detail_channels=24, gaussian_channels=128,
                 gaussian_cfg=None, graph_cfg=None, use_gaussians=True,
                 use_message_passing=True, use_context=True, use_contour=True,
                 spatial_refinement=True, coarse_loss_weight=0.3, sdf_loss_weight=0.1,
                 boundary_band_loss_weight=0.05, boundary_band_radius=3,
                 contour_truncation=5.0, gaussian_dense_weight=0.1, gaussian_node_weight=0.05,
                 num_classes=2, binary=True, ignore_index=255,
                 data_preprocessor=None, train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if num_classes < 2 or (binary and num_classes != 2):
            raise ValueError('binary v8 requires two classes; categorical v8 requires at least two')
        weights = (coarse_loss_weight, sdf_loss_weight, boundary_band_loss_weight,
                   gaussian_dense_weight, gaussian_node_weight)
        if any(value < 0 for value in weights):
            raise ValueError('loss weights must be nonnegative')
        self.num_classes = num_classes
        self.out_channels = num_classes
        self.binary = binary
        self.ignore_index = ignore_index
        self.align_corners = False
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or dict(mode='whole')
        self.use_gaussians = use_gaussians
        self.use_message_passing = use_message_passing
        self.coarse_loss_weight = coarse_loss_weight
        self.sdf_loss_weight = sdf_loss_weight
        self.boundary_band_loss_weight = boundary_band_loss_weight
        self.boundary_band_radius = boundary_band_radius
        self.contour_truncation = contour_truncation
        self.gaussian_dense_weight = gaussian_dense_weight
        self.gaussian_node_weight = gaussian_node_weight
        self.rgb_encoder = MODELS.build(rgb_encoder)
        self.register_buffer('rgb_mean', torch.tensor((123.675, 116.28, 103.53)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('rgb_std', torch.tensor((58.395, 57.12, 57.375)).view(1, 3, 1, 1), persistent=False)
        self.decoder = SpatialGeometryDecoderV7(
            rgb_channels, channels=decoder_channels, use_context=use_context,
            spatial_refinement=spatial_refinement)
        logits_channels = 1 if binary else num_classes
        self.refiner = ResidualContourHeadV6(
            decoder_channels, detail_channels, use_contour=use_contour,
            use_sdf=bool(sdf_loss_weight), num_classes=logits_channels)
        if use_gaussians:
            self.gaussian_predictor = GaussianPrimitivePredictorV8(
                rgb_channels, channels=gaussian_channels, **(gaussian_cfg or {}))
            if use_message_passing:
                self.gaussian_graph = GaussianRelationNetworkV8(
                    channels=gaussian_channels, **(graph_cfg or {}))
            self.structure_fusion = nn.ModuleList([
                GaussianFusionV8(decoder_channels, gaussian_channels) for _ in range(2)])
            self.local_fusion = nn.ModuleList([
                GaussianFusionV8(decoder_channels, gaussian_channels + 3) for _ in range(2)])
            if gaussian_dense_weight:
                self.gaussian_dense_head = nn.Conv2d(gaussian_channels + 3, logits_channels, 1)
            if gaussian_node_weight:
                self.gaussian_node_head = nn.Linear(gaussian_channels, logits_channels)

    def _normalize_bgr(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError('RPGV v8 requires exactly three BGR image channels, without pseudo geometry')
        return (inputs[:, [2, 1, 0]].float() - self.rgb_mean) / self.rgb_std

    def extract_feat(self, inputs):
        return self.rgb_encoder(self._normalize_bgr(inputs))

    def _run_network(self, inputs):
        rgb = self._normalize_bgr(inputs)
        features = self.rgb_encoder(rgb)
        gaussian = None
        if self.use_gaussians:
            primitives, raster, nodes = self.gaussian_predictor(features)
            local = GaussianSplatV8.render(nodes, raster)
            attributes = torch.cat((raster['height'], raster['coverage'], raster['variance']), dim=1)
            local = torch.cat((local, attributes), dim=1)
            graph_nodes = self.gaussian_graph(nodes, primitives, raster) if self.use_message_passing else nodes
            structure = GaussianSplatV8.render(graph_nodes, raster)
            decoded = self.decoder(
                features,
                structure_callback=lambda index, value: self.structure_fusion[index](value, structure),
                fusion_callback=lambda index, value, lateral: self.local_fusion[index](value, local))
            gaussian = dict(primitives=primitives, raster=raster, nodes=graph_nodes)
        else:
            decoded = self.decoder(features)
        outputs = self.refiner(rgb, decoded)
        outputs['gaussian'] = gaussian
        if gaussian is not None:
            if self.gaussian_dense_weight:
                outputs['gaussian_dense_logits'] = self.gaussian_dense_head(torch.cat((structure, attributes), dim=1))
            if self.gaussian_node_weight:
                outputs['gaussian_node_logits'] = self.gaussian_node_head(graph_nodes)
        logits = outputs['final_logits']
        outputs['seg_logits'] = torch.cat((-0.5 * logits, 0.5 * logits), dim=1) if self.binary else logits
        return outputs

    def loss(self, inputs, data_samples):
        labels = torch.stack([sample.gt_sem_seg.data for sample in data_samples]).to(inputs.device).long()
        if labels.shape[-2:] != inputs.shape[-2:]:
            raise ValueError('v8 expects targets padded to the input image shape')
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
        if self.sdf_loss_weight:
            losses['loss_sdf'] = self.sdf_loss_weight * sdf_loss(
                outputs['sdf'], target, valid, self.binary, self.contour_truncation)
        if self.boundary_band_loss_weight:
            losses['loss_boundary_band'] = self.boundary_band_loss_weight * boundary_band_loss(
                final, target, valid, self.binary, self.boundary_band_radius)
        if self.use_gaussians and (self.gaussian_dense_weight or self.gaussian_node_weight):
            losses.update(gaussian_semantic_losses(outputs, target, valid, self.binary,
                                                   self.gaussian_dense_weight, self.gaussian_node_weight))
        return losses

    def encode_decode(self, inputs, batch_img_metas=None):
        return F.interpolate(self._run_network(inputs)['seg_logits'], size=inputs.shape[-2:],
                             mode='bilinear', align_corners=False)

    def _forward(self, inputs, data_samples=None):
        return self.encode_decode(inputs)

    def whole_inference(self, inputs, batch_img_metas):
        return self.encode_decode(inputs, batch_img_metas)

    def slide_inference(self, inputs, batch_img_metas):
        stride_h, stride_w = self.test_cfg['stride']
        crop_h, crop_w = self.test_cfg['crop_size']
        if min(stride_h, stride_w, crop_h, crop_w) <= 0 or stride_h > crop_h or stride_w > crop_w:
            raise ValueError('slide stride must be positive and no greater than crop size')
        batch, _, height, width = inputs.shape
        predictions = inputs.new_zeros((batch, self.num_classes, height, width), dtype=torch.float32)
        counts = inputs.new_zeros((batch, 1, height, width), dtype=torch.float32)
        rows = max(height - crop_h + stride_h - 1, 0) // stride_h + 1
        columns = max(width - crop_w + stride_w - 1, 0) // stride_w + 1
        windows = {}
        for row in range(rows):
            for column in range(columns):
                y2 = min(row * stride_h + crop_h, height)
                x2 = min(column * stride_w + crop_w, width)
                y1, x1 = max(0, y2 - crop_h), max(0, x2 - crop_w)
                crop = inputs[..., y1:y2, x1:x2]
                metadata = copy.deepcopy(batch_img_metas)
                for item in metadata:
                    item['img_shape'] = crop.shape[-2:]
                logits = self.encode_decode(crop, metadata).float()
                size = logits.shape[-2:]
                if size not in windows:
                    wy = torch.hann_window(size[0], periodic=False, device=inputs.device)
                    wx = torch.hann_window(size[1], periodic=False, device=inputs.device)
                    windows[size] = torch.outer(wy, wx).clamp_min(0.05)[None, None]
                window = windows[size]
                predictions[..., y1:y2, x1:x2] += logits * window
                counts[..., y1:y2, x1:x2] += window
        return predictions / counts

    def inference(self, inputs, batch_img_metas):
        mode = self.test_cfg.get('mode', 'whole')
        if mode == 'whole':
            return self.whole_inference(inputs, batch_img_metas)
        if mode == 'slide':
            return self.slide_inference(inputs, batch_img_metas)
        raise ValueError(f'unsupported test mode: {mode}')

    def predict(self, inputs, data_samples=None):
        if data_samples is None:
            size = inputs.shape[-2:]
            metadata = [dict(ori_shape=size, img_shape=size, pad_shape=size,
                             padding_size=[0, 0, 0, 0]) for _ in range(inputs.shape[0])]
        else:
            metadata = [sample.metainfo for sample in data_samples]
        return self.postprocess_result(self.inference(inputs, metadata), data_samples)
