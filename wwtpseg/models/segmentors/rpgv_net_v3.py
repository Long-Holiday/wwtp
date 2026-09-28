"""RPGV v3: anchor the trained RGB expert and learn geometry-assisted errors."""
import math

import torch
from mmseg.registry import MODELS
from torch import nn
from torch.nn import functional as F

from .rpgv_net import _binary_segmentation_loss
from .rpgv_net_v2 import RPGVNetV2, masked_mean, region_and_boundary_loss
from ..utils.rpgv_modules import binary_entropy_from_logits
from ..utils.rpgv_v3_modules import GeometryErrorAdapter


@MODELS.register_module()
class RPGVNetV3(RPGVNetV2):
    """Train only an error adapter; frozen v2 RGB weights remain an exact anchor.

    No standalone depth-segmentation stage, deformable rectifier, Haar validator
    or feature fusion is used. Inference still uses the v2 global scene token,
    contour head and Hann sliding. initialize_rpgv_v3.py supplies the RGB anchor.
    """

    RGB_MODULES = ('rgb_encoder', 'global_film', 'rgb_aux_head',
                   'rgb_boundary_head', 'decoder', 'refiner', 'detail_refiner')
    REMOVED_MODULES = ('rectifier', 'geometry_encoder', 'geometry_pyramid',
                       'geometry_aux_head', 'frequency_validator', 'high_fusion', 'low_fusion')

    def __init__(self, adapter_channels=24, max_geometry_correction=3.0,
                 uncertainty_floor=0.2, correction_strength=1.0,
                 error_loss_weight=0.25, protection_loss_weight=0.5,
                 adapter_boundary_weight=0.1, adapter_region_weight=0.05,
                 use_geometry_evidence=True, **kwargs):
        if kwargs.get('training_stage', 'joint') != 'joint':
            raise ValueError('v3 trains an adapter on a pretrained RGB anchor; training_stage must be joint')
        for name, value in dict(uncertainty_floor=uncertainty_floor,
                                correction_strength=correction_strength).items():
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        for value in (error_loss_weight, protection_loss_weight,
                      adapter_boundary_weight, adapter_region_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError('v3 loss weights must be finite and nonnegative')
        super().__init__(**kwargs)
        for name in self.REMOVED_MODULES:
            delattr(self, name)
        channels = self.refiner.coarse_head.in_channels
        self.geometry_adapter = GeometryErrorAdapter(channels, adapter_channels, max_geometry_correction)
        self.uncertainty_floor = uncertainty_floor
        self.correction_strength = correction_strength
        self.use_geometry_evidence = use_geometry_evidence
        self.error_loss_weight = error_loss_weight
        self.protection_loss_weight = protection_loss_weight
        self.adapter_boundary_weight = adapter_boundary_weight
        self.adapter_region_weight = adapter_region_weight
        # Persistent marker prevents accidentally training/testing random frozen RGB.
        self.register_buffer('rgb_anchor_initialized', torch.tensor(False))
        self._configure_trainable_parameters()

    def _configure_trainable_parameters(self):
        if not hasattr(self, 'geometry_adapter'):
            return super()._configure_trainable_parameters()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self._set_trainable(self.geometry_adapter, True)
        for name in self.RGB_MODULES:
            getattr(self, name).eval()

    def train(self, mode=True):
        nn.Module.train(self, mode)
        for name in self.RGB_MODULES:
            getattr(self, name).eval()
        return self

    def _global_token(self, global_images):
        with torch.no_grad():
            return super()._global_token(global_images)

    def _run_network(self, inputs, global_images=None, global_token=None):
        if not bool(self.rgb_anchor_initialized):
            raise RuntimeError('v3 requires a verified RGB anchor; run tools/initialize_rpgv_v3.py first')
        # RGB heads are kept for strict checkpoint compatibility; they are not
        # auxiliary training objectives and never receive geometry gradients.
        with torch.no_grad():
            rgb, depth, quality = self._split_inputs(inputs)
            features = self._encode_rgb(inputs, global_images, global_token)
            decoded = self.decoder(features)
            base = self.refiner(rgb, decoded)
        anchor = base['final_logits'].float()
        uncertainty = binary_entropy_from_logits(anchor)
        if not self.use_geometry_evidence:
            # A capacity control uses RGB intensity and its derivatives rather
            # than all-zero descriptors that would disable trainable channels.
            depth = inputs[:, :3].float().mean(1, keepdim=True).div(255).clamp(0, 1)
            quality = torch.ones_like(depth)
        delta = self.geometry_adapter(depth, quality, decoded,
                                      anchor.sigmoid(), uncertainty)
        delta = F.interpolate(delta, size=anchor.shape[-2:], mode='bilinear', align_corners=False)
        quality = F.interpolate(quality, size=anchor.shape[-2:], mode='bilinear', align_corners=False)
        # The RGB-adapter control deliberately has no depth OR quality evidence.
        if not self.use_geometry_evidence:
            quality = torch.ones_like(quality)
        support = self.uncertainty_floor + (1 - self.uncertainty_floor) * uncertainty
        correction = self.correction_strength * quality * support * delta
        final = anchor + correction
        return dict(final_logits=final, anchor_logits=anchor,
                    geometry_correction=correction, correction_support=quality * support,
                    seg_logits=self._two_class_logits(final))

    def loss(self, inputs, data_samples):
        global_token = None
        if self.use_global_context:
            global_token = self._global_token(self._global_images_from_samples(inputs, data_samples))
        # A no-geometry adapter control must not see quality corruption/dropout.
        if self.use_geometry_evidence:
            inputs = self._apply_geometry_dropout(inputs)
        outputs = self._run_network(inputs, global_token=global_token)
        target, valid = self._targets(data_samples, inputs.shape[-2:], inputs.device)
        def resize(value):
            return F.interpolate(value.float(), size=target.shape[-2:], mode='bilinear', align_corners=False)
        final, anchor = resize(outputs['final_logits']), resize(outputs['anchor_logits'])
        losses = dict(loss_final=_binary_segmentation_loss(final, target, valid))
        error = F.binary_cross_entropy_with_logits(final, target.float(), reduction='none')
        anchor_error = F.binary_cross_entropy_with_logits(anchor, target.float(), reduction='none')
        if self.error_loss_weight:
            # Focus correction learning on RGB errors, without changing GT labels.
            focus = (anchor.sigmoid() - target).abs().detach() * valid.float()
            losses['loss_rgb_errors'] = self.error_loss_weight * masked_mean(error, focus)
        if self.protection_loss_weight:
            # Penalize worsening pixels the RGB predictor already classifies correctly.
            correct = ((anchor >= 0) == (target >= .5)) & valid
            losses['loss_rgb_protection'] = self.protection_loss_weight * masked_mean(
                F.relu(error - anchor_error.detach()), correct)
        region, boundary = region_and_boundary_loss(final, target, valid)
        if self.adapter_region_weight:
            losses['loss_region_consistency'] = self.adapter_region_weight * region
        if self.adapter_boundary_weight:
            losses['loss_final_boundary'] = self.adapter_boundary_weight * boundary
        losses['anchor_bce'] = masked_mean(anchor_error, valid).detach()
        losses['correction_abs'] = outputs['geometry_correction'].abs().mean().detach()
        return losses
