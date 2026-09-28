"""RPGV v5: full-image RGB encoding and bounded late geometry fusion."""

from __future__ import annotations

import math

import torch
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS

from .rpgv_net_v4 import RPGVNetV4
from ..utils.rpgv_v2_modules import UnifiedContourHead
from ..utils.rpgv_v5_modules import CompactGeometryEncoder, ContextTopDownDecoder


@MODELS.register_module()
class RPGVNetV5(RPGVNetV4):
    """Encode RGB once; let geometry change only the two finest decoder stages.

    ``geometry_input='rgb'`` is a pure RGB capacity control: its extra stream
    receives RGB grayscale and a constant-one gate, never depth or confidence.
    The depth model has an exact RGB fallback for three-channel or Q=0 input.
    """

    def __init__(self, rgb_encoder, rgb_channels=(64, 128, 320, 512),
                 geometry_channels=(16, 32), decoder_channels=64,
                 detail_channels=24, use_geometry=True, geometry_input='depth',
                 use_context=True, use_contour=True,
                 geometry_dropout_prob=0.1, contour_truncation=5.0,
                 max_logit_correction=2.0, coarse_loss_weight=0.3,
                 region_loss_weight=0.05, final_boundary_loss_weight=0.1,
                 sdf_loss_weight=0.1, data_preprocessor=None, train_cfg=None,
                 test_cfg=None, init_cfg=None):
        # RPGVNetV4.__init__ builds encoder-level fusion. Only BaseSegmentor's
        # constructor is needed for inherited v2 loss and inference methods.
        BaseSegmentor.__init__(self, data_preprocessor=data_preprocessor,
                               init_cfg=init_cfg)
        if len(rgb_channels) != 4 or len(geometry_channels) != 2:
            raise ValueError('v5 requires four RGB and two geometry stages')
        if geometry_input not in {'depth', 'rgb'}:
            raise ValueError("geometry_input must be 'depth' or 'rgb'")
        if not 0 <= geometry_dropout_prob <= 1:
            raise ValueError('geometry_dropout_prob must be in [0, 1]')
        if not math.isfinite(contour_truncation) or contour_truncation <= 0:
            raise ValueError('contour_truncation must be finite and positive')
        for name, value in dict(coarse_loss_weight=coarse_loss_weight,
                                region_loss_weight=region_loss_weight,
                                final_boundary_loss_weight=final_boundary_loss_weight,
                                sdf_loss_weight=sdf_loss_weight).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative')
        self.rgb_channels = tuple(rgb_channels)
        self.geometry_channels = tuple(geometry_channels)
        self.training_stage = 'joint'
        self.use_geometry = bool(use_geometry)
        self.geometry_input = geometry_input
        self.use_context = bool(use_context)
        self.use_global_context = False
        self.geometry_dropout_prob = geometry_dropout_prob if use_geometry else 0.0
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or dict(mode='whole')
        self.align_corners = False
        self.contour_truncation = contour_truncation
        self.coarse_loss_weight = coarse_loss_weight
        self.region_loss_weight = region_loss_weight
        self.final_boundary_loss_weight = final_boundary_loss_weight
        self.loss_weights = dict(boundary=0.0, sdf=sdf_loss_weight)
        self.rgb_encoder = MODELS.build(rgb_encoder)
        if (not hasattr(self.rgb_encoder, 'layers') or len(self.rgb_encoder.layers) != 4
                or tuple(self.rgb_encoder.out_indices) != (0, 1, 2, 3)):
            raise ValueError('v5 requires four-stage MixVisionTransformer with all outputs')
        if self.use_geometry:
            self.geometry_encoder = CompactGeometryEncoder(geometry_channels)
        self.decoder = ContextTopDownDecoder(
            rgb_channels, decoder_channels, geometry_channels,
            use_context=use_context, use_geometry=use_geometry)
        self.refiner = UnifiedContourHead(
            decoder_channels, detail_channels, max_logit_correction, use_contour)
        self.register_buffer(
            'rgb_mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'rgb_std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
            persistent=False)

    def extract_feat(self, inputs):
        rgb, _, _ = self._split_inputs(inputs)
        return tuple(self.rgb_encoder(rgb))

    def _apply_geometry_dropout(self, inputs):
        if self.geometry_input == 'rgb':
            # The capacity control never consumes the supplied D or Q. Its
            # dropout is applied to its synthetic all-one gate below.
            return inputs
        return super()._apply_geometry_dropout(inputs)

    def _predict_from_inputs(self, inputs, geometry_enabled=True):
        rgb, depth, confidence = self._split_inputs(inputs)
        features = tuple(self.rgb_encoder(rgb))
        geometry = None
        if self.use_geometry and geometry_enabled:
            if self.geometry_input == 'rgb':
                # BGR intensity is independent of the two geometry channels.
                depth = inputs[:, :3].float().mean(dim=1, keepdim=True).div(255).clamp(0, 1)
                confidence = torch.ones_like(depth)
                if self.training and self.geometry_dropout_prob:
                    disabled = (torch.rand(inputs.shape[0], 1, 1, 1,
                                           device=inputs.device)
                                < self.geometry_dropout_prob)
                    confidence = confidence * (~disabled).to(confidence.dtype)
            geometry = self.geometry_encoder(depth, confidence)
        decoded = self.decoder(features, geometry, confidence if geometry is not None else None)
        outputs = self.refiner(rgb, decoded)
        outputs.update(
            seg_logits=self._two_class_logits(outputs['final_logits']),
            boundary_logits=None,
            lowres_boundary_logits=None,
        )
        return outputs

    def _run_network(self, inputs, global_images=None, global_token=None):
        del global_images, global_token
        return self._predict_from_inputs(inputs, geometry_enabled=True)

    def _run_rgb_branch(self, inputs, global_images=None, global_token=None):
        del global_images, global_token
        return self._predict_from_inputs(inputs, geometry_enabled=False)
