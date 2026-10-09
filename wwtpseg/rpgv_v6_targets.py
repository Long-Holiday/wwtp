"""Exact v6 SDF targets shared by its data packer and loss fallback."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F


def contour_target_v6(target, valid, size, truncation):
    """Retain the original RPGV occupancy/observed-opposite-class EDT target."""
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
            signed = np.clip((distance_transform_edt(~neg)
                              - distance_transform_edt(~pos)) / truncation, -1, 1)
        targets.append(torch.from_numpy(signed.astype(np.float32)))
    sdf = torch.stack(targets).unsqueeze(1).to(target.device)
    radius = math.ceil(truncation)
    safe = F.max_pool2d((~known).float(), 2 * radius + 1,
                        stride=1, padding=radius) == 0
    return sdf, safe & known


def get_contour_targets_v6(target, valid, size, truncation, data_samples=None):
    """Use worker-computed targets only for the exact raster/head definition.

    Batch padding can change occupancy and the ignore-safe band. A mismatched
    cache must be recomputed from the padded labels, never resized or padded.
    """
    source_shape = tuple(target.shape[-2:])
    output_shape = tuple(size)
    if data_samples is not None and len(data_samples) == target.shape[0]:
        cached = all(
            hasattr(sample, 'gt_contour_sdf')
            and hasattr(sample, 'gt_contour_valid')
            and tuple(sample.get('v6_contour_source_shape', ())) == source_shape
            and tuple(sample.get('v6_contour_size', ())) == output_shape
            and sample.get('v6_contour_truncation') == truncation
            and tuple(sample.gt_contour_sdf.data.shape) == (1, *output_shape)
            and tuple(sample.gt_contour_valid.data.shape) == (1, *output_shape)
            for sample in data_samples)
        if cached:
            sdf = torch.stack([s.gt_contour_sdf.data for s in data_samples])
            known = torch.stack([s.gt_contour_valid.data for s in data_samples])
            return sdf.to(device=target.device, dtype=torch.float32), known.to(
                device=target.device, dtype=torch.bool)
    return contour_target_v6(target, valid, size, truncation)
