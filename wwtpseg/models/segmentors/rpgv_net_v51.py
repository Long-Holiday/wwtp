"""RPGV v5.1: optional spatial decoder residual and boundary-band loss."""

from __future__ import annotations

import math

import torch
from mmseg.registry import MODELS
from torch.nn import functional as F

from .rpgv_net_v5 import RPGVNetV5
from ..utils.rpgv_v51_modules import SpatialResidualDecoder


@MODELS.register_module()
class RPGVNetV51(RPGVNetV5):
    """Extend v5 without changing the geometry or contour interfaces.

    All existing v5 state_dict keys retain their names. New spatial blocks
    start at an exact identity, so a loaded v5 checkpoint initially predicts
    the same logits. ``boundary_mode='legacy'`` preserves v5's original shape
    losses for controlled structure-only experiments.
    """

    def __init__(self, spatial_refinement=True, boundary_mode='band',
                 boundary_band_loss_weight=0.05, boundary_band_radius=3,
                 **kwargs):
        if boundary_mode not in {'band', 'legacy'}:
            raise ValueError("boundary_mode must be 'band' or 'legacy'")
        if (not math.isfinite(boundary_band_loss_weight)
                or boundary_band_loss_weight < 0):
            raise ValueError('boundary_band_loss_weight must be finite and non-negative')
        if not isinstance(boundary_band_radius, int) or boundary_band_radius < 1:
            raise ValueError('boundary_band_radius must be a positive integer')
        self.boundary_mode = boundary_mode
        self.boundary_band_loss_weight = boundary_band_loss_weight
        self.boundary_band_radius = boundary_band_radius
        self.spatial_refinement = bool(spatial_refinement)
        kwargs = dict(kwargs)
        if boundary_mode == 'band':
            # Keep the v5 final, coarse and SDF objectives, but replace its
            # nearly constant unit-transition and region-difference terms.
            kwargs['region_loss_weight'] = 0.0
            kwargs['final_boundary_loss_weight'] = 0.0
        super().__init__(**kwargs)
        previous = self.decoder
        self.decoder = SpatialResidualDecoder(
            self.rgb_channels, previous.projections[0][0].out_channels,
            self.geometry_channels, use_context=self.use_context,
            use_geometry=self.use_geometry,
            spatial_refinement=self.spatial_refinement)
        incompatible = self.decoder.load_state_dict(previous.state_dict(), strict=False)
        expected_missing = ({key for key in incompatible.missing_keys
                             if key.startswith(('spatial8.', 'spatial4.'))}
                            if self.spatial_refinement else set())
        if (set(incompatible.missing_keys) != expected_missing
                or incompatible.unexpected_keys):
            raise RuntimeError('v5 decoder weights did not transfer to v5.1')

    def _boundary_band_loss(self, logits: torch.Tensor,
                            target: torch.Tensor,
                            valid: torch.Tensor) -> torch.Tensor:
        """Balanced foreground/background BCE within a valid GT contour band.

        Edges come only from adjacent observed GT labels. The image frame and
        ignored labels cannot create artificial boundaries. Fully empty,
        full-foreground and all-ignore masks produce a finite zero loss.
        """
        label = target.float() >= 0.5
        known = valid.bool()
        edge = torch.zeros_like(known)
        horizontal = (label[..., :, 1:] != label[..., :, :-1]) & (
            known[..., :, 1:] & known[..., :, :-1])
        vertical = (label[..., 1:, :] != label[..., :-1, :]) & (
            known[..., 1:, :] & known[..., :-1, :])
        edge[..., :, 1:] |= horizontal
        edge[..., :, :-1] |= horizontal
        edge[..., 1:, :] |= vertical
        edge[..., :-1, :] |= vertical
        radius = self.boundary_band_radius
        kernel = 2 * radius + 1
        band = F.max_pool2d(edge.float(), kernel, 1, radius) > 0
        near_ignore = F.max_pool2d((~known).float(), kernel, 1, radius) > 0
        band = band & known & ~near_ignore
        positive = band & label
        negative = band & ~label
        error = F.binary_cross_entropy_with_logits(
            logits.float(), label.float(), reduction='none')
        positive_count = positive.float().sum()
        negative_count = negative.float().sum()
        positive_loss = (error * positive.float()).sum() / positive_count.clamp_min(1)
        negative_loss = (error * negative.float()).sum() / negative_count.clamp_min(1)
        has_both = (positive_count > 0) & (negative_count > 0)
        return 0.5 * (positive_loss + negative_loss) * has_both.float()

    def _shape_losses(self, outputs, target, valid):
        losses = super()._shape_losses(outputs, target, valid)
        if self.boundary_mode == 'band' and self.boundary_band_loss_weight:
            final = F.interpolate(outputs['final_logits'], size=target.shape[-2:],
                                  mode='bilinear', align_corners=False)
            losses['loss_boundary_band'] = (
                self.boundary_band_loss_weight
                * self._boundary_band_loss(final, target, valid))
        return losses
