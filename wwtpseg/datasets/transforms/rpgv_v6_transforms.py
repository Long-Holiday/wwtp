"""V6 training packing with CPU-worker SDF target generation."""

from __future__ import annotations

import math

import torch
from mmengine.structures import PixelData
from mmseg.datasets.transforms import PackSegInputs
from mmseg.registry import TRANSFORMS

from wwtpseg.rpgv_v6_targets import contour_target_v6


@TRANSFORMS.register_module()
class PackRPGVContourInputs(PackSegInputs):
    """Pack after all geometric transforms; cache the v6 half-resolution SDF."""

    def __init__(self, contour_truncation=5.0, **kwargs):
        super().__init__(**kwargs)
        if not math.isfinite(contour_truncation) or contour_truncation <= 0:
            raise ValueError('contour_truncation must be finite and positive')
        self.contour_truncation = float(contour_truncation)

    def transform(self, results):
        packed = super().transform(results)
        sample = packed['data_samples']
        if not hasattr(sample, 'gt_sem_seg'):
            return packed
        label = sample.gt_sem_seg.data.unsqueeze(0)
        height, width = label.shape[-2:]
        size = ((height + 1) // 2, (width + 1) // 2)
        valid = label != 255
        target = torch.where(valid, label, torch.zeros_like(label))
        sdf, known = contour_target_v6(target, valid, size, self.contour_truncation)
        sample.gt_contour_sdf = PixelData(data=sdf[0])
        sample.gt_contour_valid = PixelData(data=known[0])
        sample.set_metainfo(dict(
            v6_contour_source_shape=(height, width), v6_contour_size=size,
            v6_contour_truncation=self.contour_truncation))
        return packed
