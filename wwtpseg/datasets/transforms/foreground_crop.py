"""Spatial augmentation for datasets whose positives are location-biased."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomForegroundCrop(BaseTransform):
    """Crop while moving a foreground target away from its source location.

    For a positive sample, ``foreground_prob`` controls whether a foreground-
    aware crop is used. The foreground centroid is placed at a random position
    inside the output crop, which breaks the original centre-location bias
    without discarding most positive samples. Remaining positive samples and
    all negative samples use an ordinary uniform random crop.

    Args:
        crop_size: Output height and width.
        foreground_prob: Probability of using foreground-aware sampling when
            the mask contains ``foreground_label``.
        foreground_label: Semantic label regarded as foreground.
        edge_margin_ratio: Minimum normalized distance between the sampled
            target anchor and a crop edge. Must be in ``[0, 0.5)``.
        min_foreground_pixels: Desired minimum number of foreground pixels in
            a foreground-aware crop. Small objects use their total area.
        max_attempts: Candidate crops tried before returning the one retaining
            the most foreground.
        pad_val: Constant image padding value when an input is too small.
        seg_pad_val: Constant segmentation padding value.
    """

    def __init__(
        self,
        crop_size: Sequence[int],
        foreground_prob: float = 0.8,
        foreground_label: int = 1,
        edge_margin_ratio: float = 0.15,
        min_foreground_pixels: int = 64,
        max_attempts: int = 10,
        pad_val: int = 0,
        seg_pad_val: int = 255,
    ) -> None:
        if len(crop_size) != 2 or min(crop_size) <= 0:
            raise ValueError('crop_size must contain two positive integers')
        if not 0.0 <= foreground_prob <= 1.0:
            raise ValueError('foreground_prob must be in [0, 1]')
        if not 0.0 <= edge_margin_ratio < 0.5:
            raise ValueError('edge_margin_ratio must be in [0, 0.5)')
        if min_foreground_pixels < 1 or max_attempts < 1:
            raise ValueError(
                'min_foreground_pixels and max_attempts must be positive')

        self.crop_size = tuple(int(value) for value in crop_size)
        self.foreground_prob = float(foreground_prob)
        self.foreground_label = int(foreground_label)
        self.edge_margin_ratio = float(edge_margin_ratio)
        self.min_foreground_pixels = int(min_foreground_pixels)
        self.max_attempts = int(max_attempts)
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    @staticmethod
    def _pad_array(array: np.ndarray, pad_h: int, pad_w: int,
                   value: int) -> np.ndarray:
        pad_width = [(0, pad_h), (0, pad_w)]
        pad_width.extend([(0, 0)] * (array.ndim - 2))
        return np.pad(array, pad_width, mode='constant', constant_values=value)

    def _pad_if_needed(self, results: dict) -> None:
        crop_h, crop_w = self.crop_size
        img_h, img_w = results['img'].shape[:2]
        pad_h = max(crop_h - img_h, 0)
        pad_w = max(crop_w - img_w, 0)
        if not (pad_h or pad_w):
            return

        results['img'] = self._pad_array(
            results['img'], pad_h, pad_w, self.pad_val)
        for key in results.get('seg_fields', []):
            results[key] = self._pad_array(
                results[key], pad_h, pad_w, self.seg_pad_val)

    def _uniform_bbox(self, img_h: int, img_w: int) -> tuple[int, ...]:
        crop_h, crop_w = self.crop_size
        top = np.random.randint(0, img_h - crop_h + 1)
        left = np.random.randint(0, img_w - crop_w + 1)
        return top, top + crop_h, left, left + crop_w

    def _bbox_from_anchor(self, anchor_y: float, anchor_x: float,
                          img_h: int, img_w: int) -> tuple[int, ...]:
        crop_h, crop_w = self.crop_size
        margin_y = crop_h * self.edge_margin_ratio
        margin_x = crop_w * self.edge_margin_ratio
        target_y = np.random.uniform(margin_y, crop_h - margin_y)
        target_x = np.random.uniform(margin_x, crop_w - margin_x)
        top = int(np.clip(round(anchor_y - target_y), 0, img_h - crop_h))
        left = int(np.clip(round(anchor_x - target_x), 0, img_w - crop_w))
        return top, top + crop_h, left, left + crop_w

    @staticmethod
    def _crop(array: np.ndarray, bbox: tuple[int, ...]) -> np.ndarray:
        top, bottom, left, right = bbox
        return array[top:bottom, left:right, ...]

    def _foreground_bbox(self, seg_map: np.ndarray,
                         coordinates: np.ndarray) -> tuple[int, ...]:
        img_h, img_w = seg_map.shape[:2]
        required = min(self.min_foreground_pixels, len(coordinates))
        centroid = coordinates.mean(axis=0)
        anchors = [centroid]
        if self.max_attempts > 1:
            indices = np.random.randint(
                0, len(coordinates), size=self.max_attempts - 1)
            anchors.extend(coordinates[indices])

        best_bbox = None
        best_count = -1
        for anchor_y, anchor_x in anchors:
            bbox = self._bbox_from_anchor(
                float(anchor_y), float(anchor_x), img_h, img_w)
            foreground_count = int(
                np.count_nonzero(
                    self._crop(seg_map, bbox) == self.foreground_label))
            if foreground_count > best_count:
                best_bbox = bbox
                best_count = foreground_count
            if foreground_count >= required:
                return bbox

        assert best_bbox is not None
        return best_bbox

    def transform(self, results: dict) -> dict:
        self._pad_if_needed(results)
        img_h, img_w = results['img'].shape[:2]
        seg_fields = results.get('seg_fields', [])
        seg_map = results[seg_fields[0]] if seg_fields else None
        coordinates = (
            np.argwhere(seg_map == self.foreground_label)
            if seg_map is not None else np.empty((0, 2), dtype=np.int64))
        foreground_aware = bool(
            len(coordinates)
            and np.random.random() < self.foreground_prob)

        if foreground_aware:
            bbox = self._foreground_bbox(seg_map, coordinates)
        else:
            bbox = self._uniform_bbox(img_h, img_w)

        results['img'] = self._crop(results['img'], bbox)
        for key in seg_fields:
            results[key] = self._crop(results[key], bbox)
        results['img_shape'] = results['img'].shape[:2]
        results['crop_bbox'] = bbox
        results['foreground_aware_crop'] = foreground_aware
        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(crop_size={self.crop_size}, '
            f'foreground_prob={self.foreground_prob}, '
            f'foreground_label={self.foreground_label}, '
            f'edge_margin_ratio={self.edge_margin_ratio}, '
            f'min_foreground_pixels={self.min_foreground_pixels})')
