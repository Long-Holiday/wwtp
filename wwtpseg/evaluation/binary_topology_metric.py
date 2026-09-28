"""Unfiltered mask diagnostics for islands and holes; not a topology loss."""
import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from scipy import ndimage as ndi

from .binary_boundary_metric import _extract_mask


def mask_topology(mask, valid, small_area):
    labels, count = ndi.label(mask & valid, np.ones((3, 3)))
    areas = np.bincount(labels.ravel())[1:]
    background, _ = ndi.label(~mask & valid, np.ones((3, 3)))
    exterior = np.unique(np.concatenate([
        background[0], background[-1], background[:, 0], background[:, -1],
        background[ndi.binary_dilation(~valid)],
    ]))
    holes = np.bincount(background.ravel())
    holes[exterior] = 0
    holes[0] = 0
    return dict(components=int(count), small_components=int((areas < small_area).sum()),
                holes=int((holes > 0).sum()), hole_pixels=int(holes.sum()))


@METRICS.register_module()
class BinaryTopologyMetric(BaseMetric):
    default_prefix = 'topology'

    def __init__(self, small_area=256, collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        if small_area < 1:
            raise ValueError('small_area must be positive')
        self.small_area = small_area

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            truth = _extract_mask(sample, 'gt_sem_seg')
            pred = _extract_mask(sample, 'pred_sem_seg')
            if truth.shape != pred.shape:
                raise ValueError('prediction and GT shapes must agree')
            valid = truth != 255
            gt, pred = truth == 1, (pred == 1) & valid
            labels, _ = ndi.label(pred, np.ones((3, 3)))
            areas = np.bincount(labels.ravel())
            overlap = np.bincount(labels[gt], minlength=len(areas))
            detached = overlap == 0
            detached[0] = False
            row = dict(images=1, detached_fp_components=int(detached.sum()),
                       detached_fp_pixels=int(areas[detached].sum()),
                       fp_pixels=int((pred & ~gt).sum()), negative_images=int(not gt.any()),
                       negative_images_with_fp=int(not gt.any() and pred.any()))
            for name, mask in (('gt', gt), ('pred', pred)):
                row.update({f'{name}_{k}': v for k, v in
                            mask_topology(mask, valid, self.small_area).items()})
            self.results.append(row)

    def compute_metrics(self, results):
        if not results:
            return {}
        totals = {k: sum(row[k] for row in results) for k in results[0]}
        totals['detached_fp_fraction'] = totals['detached_fp_pixels'] / max(totals['fp_pixels'], 1)
        return totals
