"""Six-class Potsdam adaptation of RPGV v2's compact contour decoder."""

import math

import torch
from mmseg.registry import MODELS
from torch import nn
from torch.nn import functional as F

from ..models.utils.rpgv_modules import ConvNormAct, DepthwiseSeparableBlock
from ..models.utils.rpgv_v2_modules import AdditiveMultiScaleDecoder
from .model_full import RPGVRemoteSensingFull


class MultiClassV2CoarseRefiner(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.coarse = nn.Conv2d(channels, num_classes, 1)
        self.boundary = nn.Conv2d(channels, 1, 1)

    def forward(self, feature: torch.Tensor, reliability=None) -> dict:
        del reliability
        coarse = self.coarse(feature)
        return dict(final=coarse, coarse=coarse,
                    boundary=self.boundary(feature),
                    refinement_gate=torch.ones_like(coarse[:, :1]))


class MultiClassV2ContourRefiner(nn.Module):
    """One half-resolution contour field gives bounded, class-wise logit edits."""

    def __init__(self, channels: int, num_classes: int,
                 detail_channels: int = 24, max_logit_correction: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.max_logit_correction = max_logit_correction
        self.rgb_stem = nn.Sequential(
            ConvNormAct(3, detail_channels),
            DepthwiseSeparableBlock(detail_channels, detail_channels))
        self.semantic_projection = ConvNormAct(channels, detail_channels,
                                                kernel_size=1)
        self.contour_features = nn.Sequential(
            ConvNormAct(2 * detail_channels + num_classes, detail_channels,
                        kernel_size=1),
            DepthwiseSeparableBlock(detail_channels, detail_channels))
        self.class_correction = nn.Conv2d(detail_channels, num_classes, 1)
        self.boundary = nn.Conv2d(detail_channels, 1, 1)
        nn.init.zeros_(self.class_correction.weight)
        nn.init.zeros_(self.class_correction.bias)

    def forward(self, rgb: torch.Tensor, decoder_feature: torch.Tensor,
                base_logits: torch.Tensor, reliability=None) -> dict:
        del reliability
        detail = self.rgb_stem(F.avg_pool2d(rgb.float(), 2, stride=2,
                                            ceil_mode=True))
        size = detail.shape[-2:]
        base = F.interpolate(base_logits, size=size, mode='bilinear',
                             align_corners=False)
        semantic = F.interpolate(self.semantic_projection(decoder_feature),
                                 size=size, mode='bilinear', align_corners=False)
        features = self.contour_features(torch.cat(
            [detail, semantic, base.softmax(dim=1)], dim=1))
        probabilities = base.float().softmax(dim=1)
        uncertainty = -(probabilities * (probabilities + 1e-7).log()).sum(
            dim=1, keepdim=True) / math.log(self.num_classes)
        correction = (self.max_logit_correction * uncertainty.detach()
                      * self.class_correction(features).float().tanh())
        return dict(final=base.float() + correction,
                    boundary=self.boundary(features),
                    refinement_gate=uncertainty)


@MODELS.register_module()
class RPGVRemoteSensingV2(RPGVRemoteSensingFull):
    """RPGV v2 for six land-cover classes, sharing one architecture for both depth sources.

    The RGR, geometry validator, global context and reliability fallback are
    inherited from the complete multiclass RPGV model. The additive decoder
    and bounded uncertainty-guided contour correction follow RPGVNetV2.
    """

    def __init__(self, max_logit_correction: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.decoder = AdditiveMultiScaleDecoder(self.rgb_channels,
                                                 self.decoder_channels)
        self.refiner = MultiClassV2CoarseRefiner(self.decoder_channels,
                                                 self.num_classes)
        self.detail_refiner = MultiClassV2ContourRefiner(
            self.decoder_channels, self.num_classes, self.detail_channels,
            max_logit_correction)
