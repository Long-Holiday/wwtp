"""Foreground, overlap and boundary metrics for binary segmentation."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from scipy.ndimage import binary_erosion, distance_transform_edt


def _extract_mask(sample, key: str) -> np.ndarray:
    if isinstance(sample, dict):
        val = sample[key]
        if isinstance(val, dict):
            val = val['data']
        elif hasattr(val, 'data'):
            val = val.data
    else:
        val = getattr(sample, key)
        if hasattr(val, 'data'):
            val = val.data
    if torch.is_tensor(val):
        val = val.squeeze().detach().cpu().numpy()
    elif hasattr(val, 'numpy'):
        val = val.squeeze().numpy()
    else:
        val = np.asarray(val).squeeze()
    return val


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Return the one-pixel inner boundary of a binary mask."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return np.logical_xor(mask, eroded)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


@METRICS.register_module()
class BinaryBoundaryMetric(BaseMetric):
    """Evaluate binary foreground and localization quality.

    Boundary F1 follows tolerance-based contour matching. Hausdorff distance
    is computed symmetrically from boundary pixels without materializing the
    quadratic pairwise distance matrix. If exactly one mask is empty, the
    image diagonal is used as a finite worst-case penalty; if both are empty,
    the distance is zero. ``HD95`` is included because ordinary Hausdorff
    distance is very sensitive to a single outlier.

    Args:
        boundary_tolerance: Matching radius in pixels for Boundary F1.
        pixel_size_m: Ground sampling distance, used for metre-valued HD.
        collect_device: MMEngine distributed result collection device.
        prefix: Metric key prefix.
    """

    default_prefix = 'binary'

    def __init__(
        self,
        boundary_tolerance: float = 3.0,
        pixel_size_m: float = 0.5,
        collect_device: str = 'cpu',
        prefix: str | None = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if boundary_tolerance < 0:
            raise ValueError('boundary_tolerance must be non-negative')
        if pixel_size_m <= 0:
            raise ValueError('pixel_size_m must be positive')
        self.boundary_tolerance = float(boundary_tolerance)
        self.pixel_size_m = float(pixel_size_m)
        self._target_cache: dict = {}

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Collect per-image sufficient statistics."""
        del data_batch
        for sample in data_samples:
            pred = _extract_mask(sample, 'pred_sem_seg')
            target = _extract_mask(sample, 'gt_sem_seg')
            valid = target != 255
            pred = np.logical_and(pred == 1, valid)
            target = np.logical_and(target == 1, valid)

            tp = np.logical_and(pred, target).sum(dtype=np.int64)
            fp = np.logical_and(pred, np.logical_not(target)).sum(dtype=np.int64)
            fn = np.logical_and(np.logical_not(pred), target).sum(dtype=np.int64)
            tn = np.logical_and(np.logical_not(pred), np.logical_not(target))
            tn = np.logical_and(tn, valid).sum(dtype=np.int64)

            pred_boundary = _boundary(pred)
            pred_count = int(pred_boundary.sum())

            img_path = sample.get('img_path', None) if isinstance(sample, dict) else getattr(sample, 'img_path', None)
            if img_path and img_path in self._target_cache:
                target_boundary, target_count, distance_to_target = self._target_cache[img_path]
            else:
                target_boundary = _boundary(target)
                target_count = int(target_boundary.sum())
                distance_to_target = distance_transform_edt(~target_boundary) if target_count else None
                if img_path:
                    self._target_cache[img_path] = (target_boundary, target_count, distance_to_target)

            matched_pred = 0
            matched_target = 0
            distance_to_pred = None

            if pred_count and target_count:
                matched_pred = int(
                    np.logical_and(
                        pred_boundary,
                        distance_to_target <= self.boundary_tolerance,
                    ).sum())
                distance_to_pred = distance_transform_edt(~pred_boundary)
                matched_target = int(
                    np.logical_and(
                        target_boundary,
                        distance_to_pred <= self.boundary_tolerance,
                    ).sum())

            hd, hd95 = self._hausdorff(
                pred_boundary, target_boundary, distance_to_target,
                distance_to_pred)
            self.results.append(
                dict(
                    tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                    pred_boundary=pred_count,
                    target_boundary=target_count,
                    matched_pred=matched_pred,
                    matched_target=matched_target,
                    hd=hd,
                    hd95=hd95,
                ))

    @staticmethod
    def _hausdorff(
        pred_boundary: np.ndarray,
        target_boundary: np.ndarray,
        distance_to_target: np.ndarray | None,
        distance_to_pred: np.ndarray | None,
    ) -> tuple[float, float]:
        pred_count = int(pred_boundary.sum())
        target_count = int(target_boundary.sum())
        if pred_count == 0 and target_count == 0:
            return 0.0, 0.0
        if pred_count == 0 or target_count == 0:
            height, width = pred_boundary.shape
            penalty = float(np.hypot(height - 1, width - 1))
            return penalty, penalty

        if distance_to_target is None:
            distance_to_target = distance_transform_edt(~target_boundary)
        if distance_to_pred is None:
            distance_to_pred = distance_transform_edt(~pred_boundary)
        distances = np.concatenate([
            distance_to_target[pred_boundary],
            distance_to_pred[target_boundary],
        ])
        return float(distances.max()), float(np.percentile(distances, 95))

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Aggregate global overlap/BF statistics and image-mean HD."""
        totals = {
            key: sum(item[key] for item in results)
            for key in (
                'tp', 'fp', 'fn', 'tn', 'pred_boundary', 'target_boundary',
                'matched_pred', 'matched_target'
            )
        }
        tp, fp, fn = totals['tp'], totals['fp'], totals['fn']
        foreground_iou = _safe_div(tp, tp + fp + fn)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        dice = _safe_div(2 * tp, 2 * tp + fp + fn)

        boundary_precision = _safe_div(
            totals['matched_pred'], totals['pred_boundary'])
        boundary_recall = _safe_div(
            totals['matched_target'], totals['target_boundary'])
        boundary_f1 = _safe_div(
            2 * boundary_precision * boundary_recall,
            boundary_precision + boundary_recall,
        )

        hd = float(np.mean([item['hd'] for item in results]))
        hd95 = float(np.mean([item['hd95'] for item in results]))
        return {
            'Foreground_IoU': 100.0 * foreground_iou,
            'Dice': 100.0 * dice,
            'F1': 100.0 * dice,
            'Precision': 100.0 * precision,
            'Recall': 100.0 * recall,
            'Boundary_F1': 100.0 * boundary_f1,
            'BFScore': 100.0 * boundary_f1,
            'Boundary_Precision': 100.0 * boundary_precision,
            'Boundary_Recall': 100.0 * boundary_recall,
            'Hausdorff_px': hd,
            'HD95_px': hd95,
            'Hausdorff_m': hd * self.pixel_size_m,
            'HD95_m': hd95 * self.pixel_size_m,
        }
