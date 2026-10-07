"""Reject accidental use of old full-resolution WWTP data."""
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RequireImageShape(BaseTransform):
    def __init__(self, size=(1024, 1024)):
        self.size = tuple(size)

    def transform(self, results):
        if results['img'].shape[:2] != self.size:
            raise ValueError(
                f"{results['img_path']}: expected {self.size}, got "
                f"{results['img'].shape[:2]}. Prepare the 1024 dataset first.")
        if 'gt_seg_map' in results and results['gt_seg_map'].shape != self.size:
            raise ValueError(f"{results['img_path']}: image and annotation shapes differ")
        return results
