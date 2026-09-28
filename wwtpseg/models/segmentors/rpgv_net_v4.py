"""Single-stage RPGV v4: geometry fusion inside the v2 MiT encoder."""
import math

import torch
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from torch import nn
from torch.nn import functional as F

from .rpgv_net import _binary_segmentation_loss
from .rpgv_net_v2 import RPGVNetV2
from ..utils.rpgv_modules import ConvNormAct, DepthwiseSeparableBlock, GlobalContextFiLM
from ..utils.rpgv_v2_modules import AdditiveMultiScaleDecoder, UnifiedContourHead
from ..utils.rpgv_v4_modules import GeometryFusionBlock


@MODELS.register_module()
class RPGVNetV4(RPGVNetV2):
    """Reuse v2 losses/head and inference, without constructing v1/v2 branches.

    All parameters train together from iteration one. MiT stages consume the
    preceding fused RGB features; a compact geometry stream also receives RGB
    context. ImageNet MiT initialization remains usable without staged weights.
    """

    def __init__(self, rgb_encoder, rgb_channels=(64, 128, 320, 512),
                 geometry_channels=(16, 32, 64, 128), decoder_channels=64,
                 detail_channels=24, use_geometry=True, use_frequency=True,
                 context_grid=4, use_global_context=False, global_thumbnail_size=512,
                 geometry_dropout_prob=0.1, use_contour=True,
                 max_logit_correction=2.0, contour_truncation=10.0,
                 coarse_loss_weight=0.3, region_loss_weight=0.1,
                 final_boundary_loss_weight=0.1, sdf_loss_weight=0.1,
                 data_preprocessor=None, train_cfg=None, test_cfg=None, init_cfg=None):
        # Deliberately bypass RPGVNetV2.__init__: removed modules must not occupy
        # memory, optimizer state, or checkpoints as dormant/frozen branches.
        BaseSegmentor.__init__(self, data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if len(rgb_channels) != 4 or len(geometry_channels) != 4:
            raise ValueError('v4 requires four RGB and geometry stages')
        if not 0 <= geometry_dropout_prob <= 1:
            raise ValueError('geometry_dropout_prob must be in [0, 1]')
        if not isinstance(context_grid, int) or context_grid < 0:
            raise ValueError('context_grid must be a non-negative integer')
        if contour_truncation <= 0 or global_thumbnail_size <= 0:
            raise ValueError('contour truncation and thumbnail size must be positive')
        for weight in (coarse_loss_weight, region_loss_weight,
                       final_boundary_loss_weight, sdf_loss_weight):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError('loss weights must be finite and non-negative')
        if not use_contour and sdf_loss_weight == 0:
            raise ValueError('disabled contour correction still requires SDF supervision')
        self.rgb_channels = tuple(rgb_channels)
        self.geometry_channels = tuple(geometry_channels)
        self.training_stage = 'joint'  # inherited dropout/inference dispatch only
        self.use_geometry = use_geometry
        self.use_frequency = use_frequency
        self.use_global_context = use_global_context
        self.global_thumbnail_size = global_thumbnail_size
        self.geometry_dropout_prob = geometry_dropout_prob if use_geometry else 0.0
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or dict(mode='whole')
        self.align_corners = False
        self.contour_truncation = contour_truncation
        self.coarse_loss_weight = coarse_loss_weight
        self.region_loss_weight = region_loss_weight
        self.final_boundary_loss_weight = final_boundary_loss_weight
        # Reuse v2 shape supervision, with no independent RGB boundary head.
        self.loss_weights = dict(boundary=0.0, sdf=sdf_loss_weight)
        self.rgb_encoder = MODELS.build(rgb_encoder)
        if (not hasattr(self.rgb_encoder, 'layers') or len(self.rgb_encoder.layers) != 4
                or tuple(self.rgb_encoder.out_indices) != (0, 1, 2, 3)):
            raise ValueError('v4 requires a four-stage MixVisionTransformer with all outputs')
        if use_global_context:
            self.global_film = GlobalContextFiLM(rgb_channels[-1], rgb_channels)
        if use_geometry:
            self.geometry_downsamples = nn.ModuleList([
                nn.Sequential(ConvNormAct(2, geometry_channels[0], stride=2),
                              DepthwiseSeparableBlock(geometry_channels[0], geometry_channels[0], stride=2)),
                *[DepthwiseSeparableBlock(geometry_channels[i - 1], geometry_channels[i], stride=2)
                  for i in range(1, 4)]])
            self.geometry_fusions = nn.ModuleList([
                GeometryFusionBlock(r, g, use_frequency=use_frequency and i < 2,
                                    context_grid=context_grid if i >= 2 else 0,
                                    update_geometry=i < 3)
                for i, (r, g) in enumerate(zip(rgb_channels, geometry_channels))])
        self.decoder = AdditiveMultiScaleDecoder(rgb_channels, decoder_channels)
        self.refiner = UnifiedContourHead(decoder_channels, detail_channels,
                                         max_logit_correction, use_contour)
        self.register_buffer('rgb_mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('rgb_std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1), persistent=False)

    def _split_inputs(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] not in (3, 5):
            raise ValueError('v4 expects BGR or BGR + depth + confidence (3 or 5 channels)')
        if inputs.shape[1] == 3:
            inputs = torch.cat([inputs, inputs.new_zeros(inputs.shape[0], 2, *inputs.shape[-2:])], 1)
        return super()._split_inputs(inputs)

    def _encode_fused(self, inputs, global_images=None, global_token=None, geometry_enabled=True):
        rgb, depth, confidence = self._split_inputs(inputs)
        if not geometry_enabled:
            confidence = torch.zeros_like(confidence)
        if self.use_global_context and global_token is None:
            if global_images is None:
                global_images = self._global_images_from_samples(inputs, None)
            global_token = self._global_token(global_images)
        # Mask before any spatial mixing: arbitrary depth at Q=0 cannot leak
        # into a reliable neighbour through convolutions or Haar transforms.
        geometry = torch.cat([depth * confidence, confidence], dim=1)
        x, features = rgb, []
        for i, layer in enumerate(self.rgb_encoder.layers):
            tokens, shape = layer[0](x)
            for block in layer[1]:
                tokens = block(tokens, shape)
            tokens = layer[2](tokens)
            x = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, *shape).contiguous()
            if self.use_global_context:
                gamma, beta = self.global_film.modulators[i](global_token).chunk(2, dim=1)
                x = x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
            if self.use_geometry:
                geometry = self.geometry_downsamples[i](geometry)
                if geometry.shape[-2:] != shape:
                    geometry = F.interpolate(geometry, size=shape, mode='bilinear', align_corners=False)
                x, geometry = self.geometry_fusions[i](x, geometry, confidence)
            features.append(x)
        return tuple(features)

    def extract_feat(self, inputs):
        return self._encode_fused(inputs)

    def _run_network(self, inputs, global_images=None, global_token=None):
        features = self._encode_fused(inputs, global_images, global_token)
        rgb, _, _ = self._split_inputs(inputs)
        outputs = self._decode_and_refine(rgb, features, None)
        outputs['fused_features'] = features
        return outputs

    def _run_rgb_branch(self, inputs, global_images=None, global_token=None):
        features = self._encode_fused(inputs, global_images, global_token, geometry_enabled=False)
        rgb, _, _ = self._split_inputs(inputs)
        return self._decode_and_refine(rgb, features, None)

    def loss(self, inputs, data_samples):
        if inputs.shape[1] == 5:
            inputs = self._apply_geometry_dropout(inputs)
        global_images = (self._global_images_from_samples(inputs, data_samples)
                         if self.use_global_context else None)
        outputs = self._run_network(inputs, global_images=global_images)
        target, valid = self._targets(data_samples, inputs.shape[-2:], inputs.device)
        logits = F.interpolate(outputs['final_logits'], size=target.shape[-2:],
                               mode='bilinear', align_corners=False)
        losses = dict(loss_final=_binary_segmentation_loss(logits, target, valid))
        losses.update(self._shape_losses(outputs, target, valid))
        return losses
