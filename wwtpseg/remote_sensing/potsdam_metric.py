"""Potsdam benchmark metrics: five scored classes, optional 3 px boundary erosion."""

import cv2
import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class PotsdamFiveClassMetric(BaseMetric):
    """Accumulate global pixel confusion after excluding clutter and ignored pixels.

    ``erode_radius=3`` approximates the ISPRS eroded-reference protocol by
    retaining only pixels 3 px inside each class region. The official eroded
    reference files are preferable for exact leaderboard replication.
    """

    def __init__(self, erode_radius=0, collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.erode_radius = erode_radius
        self.kernel = (cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_radius + 1, 2 * erode_radius + 1))
            if erode_radius else None)

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            pred_field = sample['pred_sem_seg'] if isinstance(sample, dict) else sample.pred_sem_seg
            gt_field = sample['gt_sem_seg'] if isinstance(sample, dict) else sample.gt_sem_seg
            pred_data = pred_field['data'] if isinstance(pred_field, dict) else pred_field.data
            gt_data = gt_field['data'] if isinstance(gt_field, dict) else gt_field.data
            pred = pred_data.squeeze().cpu().numpy().astype(np.int64)
            gt = gt_data.squeeze().cpu().numpy().astype(np.int64)
            valid = (gt >= 0) & (gt < 5) & (pred >= 0) & (pred < 6)
            if self.kernel is not None:
                interiors = np.zeros(gt.shape, dtype=bool)
                for class_id in range(5):
                    interiors |= cv2.erode((gt == class_id).astype(np.uint8), self.kernel).astype(bool)
                valid &= interiors
            # Prediction 5 (clutter) counts as a false negative for the true class.
            matrix = np.bincount(gt[valid] * 6 + pred[valid], minlength=30).reshape(5, 6)
            self.results.append(matrix)

    def compute_metrics(self, results):
        matrix = np.sum(results, axis=0, dtype=np.int64)
        tp = matrix[:, :5].diagonal().astype(np.float64)
        gt_count = matrix.sum(axis=1).astype(np.float64)
        pred_count = matrix[:, :5].sum(axis=0).astype(np.float64)
        union = gt_count + pred_count - tp
        denom_f1 = gt_count + pred_count
        iou = np.divide(tp, union, out=np.full(5, np.nan), where=union > 0)
        f1 = np.divide(2 * tp, denom_f1, out=np.full(5, np.nan), where=denom_f1 > 0)
        names = ('impervious', 'building', 'low_vegetation', 'tree', 'car')
        metrics = {
            'mIoU5': round(float(np.nanmean(iou) * 100), 2),
            'mF15': round(float(np.nanmean(f1) * 100), 2),
            'OA5': round(float(tp.sum() / max(gt_count.sum(), 1) * 100), 2),
        }
        for i, name in enumerate(names):
            metrics[f'IoU_{name}'] = round(float(iou[i] * 100), 2)
            metrics[f'F1_{name}'] = round(float(f1[i] * 100), 2)
        return metrics
