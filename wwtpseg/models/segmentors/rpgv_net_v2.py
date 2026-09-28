"""WWTP RPGV v2: retain validated geometry, simplify and supervise contours."""
import math

import numpy as np
import torch
from mmseg.registry import MODELS
from torch import nn
from torch.nn import functional as F

from .rpgv_net import RPGVNet, _binary_segmentation_loss, _boundary_target
from ..utils.rpgv_modules import MultiScaleDecoder
from ..utils.rpgv_v2_modules import AdditiveMultiScaleDecoder, UnifiedContourHead


def masked_mean(value, mask):
    return (value.float() * mask.float()).sum() / mask.float().sum().clamp_min(1)


def region_and_boundary_loss(logits, target, valid):
    """Supervise final probabilities, balancing same-class and contour pairs.

    Region consistency only couples pixels whose GT classes agree. Boundary
    pairs preserve the signed transition, so smoothing cannot earn a lower
    contour loss by erasing a true narrow object. No connected-component prior.
    """
    probability = logits.float().sigmoid()
    region, boundary = probability.sum() * 0, probability.sum() * 0
    for dimension in (-2, -1):
        left = [slice(None)] * 4
        right = [slice(None)] * 4
        left[dimension], right[dimension] = slice(None, -1), slice(1, None)
        a, b = tuple(left), tuple(right)
        delta = probability[b] - probability[a]
        truth = target[b].float() - target[a].float()
        pairs = valid[a] & valid[b]
        same = pairs & (truth == 0)
        different = pairs & (truth != 0)
        region = region + masked_mean(delta.abs(), same)
        boundary = boundary + masked_mean((delta - truth).abs(), different)
    return region / 2, boundary / 2


def contour_target(target, valid, size, truncation):
    """Half-resolution EDT with conservative masking around unknown labels.

    Uniform foreground/background maps have constant +/-1 distance targets;
    crop borders and ignore labels are not treated as observed class edges.
    """
    from scipy.ndimage import distance_transform_edt

    soft = F.adaptive_avg_pool2d(target.float(), size)
    known = F.adaptive_avg_pool2d(valid.float(), size) >= 1.0
    mask = soft >= 0.5
    targets = []
    for fg, ok in zip(mask.detach().cpu().numpy(), known.detach().cpu().numpy()):
        fg, ok = fg[0], ok[0]
        pos, neg = fg & ok, ~fg & ok
        if not pos.any():
            signed = np.full(fg.shape, -1, dtype=np.float32)
        elif not neg.any():
            signed = np.ones(fg.shape, dtype=np.float32)
        else:
            # Distance to an observed opposite class, never to unknown pixels.
            signed = np.clip((distance_transform_edt(~neg)
                              - distance_transform_edt(~pos)) / truncation, -1, 1)
        targets.append(torch.from_numpy(signed.astype(np.float32)))
    sdf = torch.stack(targets).unsqueeze(1).to(target.device)
    radius = math.ceil(truncation)
    safe = F.max_pool2d((~known).float(), 2 * radius + 1,
                        stride=1, padding=radius) == 0
    return sdf, safe & known


@MODELS.register_module()
class RPGVNetV2(RPGVNet):
    """V1 upstream state keys stay compatible; prediction heads are new.

    Standard training uses joint optimization from initialization (see
    configs/experiments/rpgv_v2_joint.py). Legacy stage modes remain available
    for historical checkpoint diagnostics. Corruption supervision, Q=0 fallback
    and Hann sliding inference are inherited; no staged weights are required.
    """

    def __init__(self, decoder_channels=64, detail_channels=24,
                 contour_truncation=10.0, max_logit_correction=2.0,
                 coarse_loss_weight=0.3, region_loss_weight=0.1,
                 correction_smoothness_weight=0.02, use_contour=True,
                 final_boundary_loss_weight=None, decoder_type='additive',
                 use_uncertainty_gate=True, learnable_loss_weights=False, **kwargs):
        if contour_truncation <= 0:
            raise ValueError('contour_truncation must be positive')
        for weight in (coarse_loss_weight, region_loss_weight, correction_smoothness_weight):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError('v2 loss weights must be finite and non-negative')
        if decoder_type not in {'additive', 'dense'}:
            raise ValueError('decoder_type must be additive or dense')
        if final_boundary_loss_weight is not None and (
            not math.isfinite(final_boundary_loss_weight) or final_boundary_loss_weight < 0
        ):
            raise ValueError('final_boundary_loss_weight must be finite and non-negative')
        # Old serial-head switches have no meaning for this unified head.
        components = kwargs.get('component_cfg') or {}
        if any(components.get(k) is False for k in ('boundary_refinement', 'detail_refinement')):
            raise ValueError('v2 uses use_contour=False to ablate its unified contour correction')
        self._contour_options = dict(max_logit_correction=max_logit_correction,
                                     use_contour=use_contour,
                                     use_uncertainty_gate=use_uncertainty_gate)
        self.decoder_type = decoder_type
        super().__init__(decoder_channels=decoder_channels,
                         detail_channels=detail_channels, **kwargs)
        self.contour_truncation = contour_truncation
        self.coarse_loss_weight = coarse_loss_weight
        self.region_loss_weight = region_loss_weight
        self.correction_smoothness_weight = correction_smoothness_weight
        # None preserves the previous coefficient, including user overrides.
        self.final_boundary_loss_weight = (
            self.loss_weights['boundary'] / 2 if final_boundary_loss_weight is None
            else final_boundary_loss_weight)
        self.learnable_loss_weights = learnable_loss_weights
        if learnable_loss_weights:
            if self.training_stage != 'joint':
                raise ValueError('learnable loss weights require single-stage joint training')
            # Existing coefficients specify initialization and zero disables a
            # term. Unscale inherited losses before applying learned weights.
            initial = {f'loss_{name}': value for name, value in self.loss_weights.items()}
            initial.update(
                loss_boundary=self.loss_weights['boundary'] / 2,
                loss_coarse=coarse_loss_weight,
                loss_region_consistency=region_loss_weight,
                loss_final_boundary=self.final_boundary_loss_weight,
                loss_correction_smoothness=correction_smoothness_weight)
            if not self.component_enabled('depth_rectification'):
                for name in ('preserve', 'equivariance', 'correction_smoothness'):
                    initial.pop(f'loss_{name}')
            if not self.component_enabled('learned_reliability'):
                initial.pop('loss_reliability')
            if any(not math.isfinite(v) or v < 0 for v in initial.values()):
                raise ValueError('loss weight initializers must be finite and non-negative')
            self._loss_initial_weights = {k: v for k, v in initial.items() if v > 0}
            self.loss_log_variances = nn.ParameterDict({
                k: nn.Parameter(torch.tensor(-math.log(v), dtype=torch.float32))
                for k, v in self._loss_initial_weights.items()})

    def loss(self, inputs, data_samples):
        losses = super().loss(inputs, data_samples)
        if not self.learnable_loss_weights:
            return losses
        return self._weight_losses(losses)

    def _weight_losses(self, losses):
        """Bounded uncertainty weighting: exp(-s_i) * raw_loss_i + s_i.

        The +s term prevents driving all weights to zero. Bounding s to [-6, 6]
        also bounds weights for auxiliary losses that can approach zero. The
        objective can be negative; checkpoint selection still uses val IoU.
        Logging keys intentionally exclude 'loss' (MMEngine sums those keys).
        """
        result = dict(losses)
        for name, parameter in self.loss_log_variances.items():
            raw = losses[name].float() / self._loss_initial_weights[name]
            log_variance = parameter.float().clamp(-6.0, 6.0)
            weight = torch.exp(-log_variance)
            result[name] = weight * raw + log_variance
            label = name.removeprefix('loss_')
            result[f'weight/{label}'] = weight.detach()
            result[f'raw/{label}'] = raw.detach()
        return result

    def _build_prediction_modules(self, decoder_channels, detail_channels):
        decoder = AdditiveMultiScaleDecoder if self.decoder_type == 'additive' else MultiScaleDecoder
        self.decoder = decoder(self.rgb_channels, decoder_channels)
        self.refiner = UnifiedContourHead(decoder_channels, detail_channels,
                                         **self._contour_options)
        # Parameter-free placeholder preserves the inherited stage-management API.
        self.detail_refiner = nn.Identity()

    def _decode_and_refine(self, rgb, features, reliability):
        outputs = self.refiner(rgb, self.decoder(features))
        outputs.update(seg_logits=self._two_class_logits(outputs['final_logits']),
                       boundary_logits=None, lowres_boundary_logits=None)
        return outputs

    def _shape_losses(self, outputs, target, valid):
        losses = {}
        coarse = outputs['coarse_logits']
        # Fractional occupancy retains thin/small GT regions in coarse supervision.
        size = coarse.shape[-2:]
        coarse_valid = F.adaptive_avg_pool2d(valid.float(), size) >= 1.0
        coarse_target = F.adaptive_avg_pool2d(target.float(), size)
        if self.coarse_loss_weight:
            losses['loss_coarse'] = self.coarse_loss_weight * _binary_segmentation_loss(
                coarse, coarse_target, coarse_valid)
        final = F.interpolate(outputs['final_logits'], size=target.shape[-2:],
                              mode='bilinear', align_corners=False)
        region, boundary = region_and_boundary_loss(final, target, valid)
        if self.region_loss_weight:
            losses['loss_region_consistency'] = self.region_loss_weight * region
        if self.final_boundary_loss_weight:
            losses['loss_final_boundary'] = self.final_boundary_loss_weight * boundary
        if self.loss_weights['boundary']:
            # Keep the supervised RGB boundary cue used by RGR, but make final
            # output contours (not only auxiliary heads) pay the boundary cost.
            prediction = outputs['rgb_boundary_logits']
            mask, known = self._resize_targets(target, valid, prediction.shape[-2:])
            known = F.max_pool2d((~known).float(), 3, 1, 1) == 0
            aux = _binary_segmentation_loss(prediction, _boundary_target(mask, known), known)
            losses['loss_boundary'] = self.loss_weights['boundary'] * aux / 2
        if self.loss_weights['sdf']:
            sdf, known = contour_target(target, valid, outputs['sdf'].shape[-2:],
                                        self.contour_truncation)
            error = F.smooth_l1_loss(outputs['sdf'].float().tanh(), sdf, reduction='none')
            # Emphasize the contour band while still supervising region sign.
            weight = 1 + 4 * (1 - sdf.abs())
            losses['loss_sdf'] = self.loss_weights['sdf'] * masked_mean(error * weight, known)
        return losses

    def _geometry_losses(self, inputs, global_token, outputs, target, valid, data_samples):
        losses = super()._geometry_losses(inputs, global_token, outputs,
                                          target, valid, data_samples)
        if self.correction_smoothness_weight and self.component_enabled('depth_rectification'):
            base = outputs['base_depth'].float()
            correction = outputs['corrected_depth'].float() - base
            _, known = self._resize_targets(target, valid, base.shape[-2:])
            penalty = correction.sum() * 0
            for dim in (-2, -1):
                a, b = [slice(None)] * 4, [slice(None)] * 4
                a[dim], b[dim] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                weight = torch.exp(-10 * (base[b] - base[a]).abs())
                penalty = penalty + masked_mean(
                    (correction[b] - correction[a]).abs() * weight, known[a] & known[b])
            losses['loss_correction_smoothness'] = self.correction_smoothness_weight * penalty / 2
        return losses
