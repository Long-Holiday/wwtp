"""Data transforms used by the staged RPGV-Net training pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import PixelData
from mmseg.datasets.transforms import PackSegInputs
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GenerateGlobalThumbnail(BaseTransform):
    """Capture an RGB-only scene thumbnail before local spatial augmentation.

    The transform must run after photometric augmentation and before pseudo
    geometry is appended.  The thumbnail is packed separately and therefore
    is not cropped or rotated with the local training patch.
    """

    def __init__(self, size: tuple[int, int] = (512, 512)) -> None:
        self.size = tuple(size)
        if len(self.size) != 2 or min(self.size) <= 0:
            raise ValueError('thumbnail size must contain two positive values')

    def transform(self, results: dict) -> dict:
        image = results['img']
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                'GenerateGlobalThumbnail expects a three-channel RGB/BGR '
                f'image, got {image.shape}')
        height, width = self.size
        results['global_img'] = cv2.resize(
            image, (width, height), interpolation=cv2.INTER_AREA)
        return results


@TRANSFORMS.register_module()
class RandomPseudoGeometryCorruption(BaseTransform):
    """Apply one synthetic failure mode to pseudo depth during training.

    Input reliability is deliberately kept unchanged.  A training-only soft
    validity target records the affected area/severity, but it is packed in
    the data sample rather than appended to the model input.  The learned
    reliability must therefore identify geometry that is inconsistent with
    RGB instead of reading an artificial corruption flag.
    """

    MODES = ('noise', 'blur', 'block', 'scale_shift', 'zero')

    def __init__(
        self,
        prob: float = 0.3,
        modes: tuple[str, ...] = MODES,
        noise_std_range: tuple[float, float] = (0.02, 0.1),
        block_ratio_range: tuple[float, float] = (0.1, 0.3),
        scale_range: tuple[float, float] = (0.7, 1.3),
        shift_range: tuple[float, float] = (-0.15, 0.15),
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError('prob must lie in [0, 1]')
        if not modes:
            raise ValueError('at least one corruption mode is required')
        unknown = set(modes) - set(self.MODES)
        if unknown:
            raise ValueError(f'unknown pseudo-geometry corruptions: {unknown}')
        for name, value_range in (
            ('noise_std_range', noise_std_range),
            ('block_ratio_range', block_ratio_range),
            ('scale_range', scale_range),
            ('shift_range', shift_range),
        ):
            if len(value_range) != 2 or value_range[0] > value_range[1]:
                raise ValueError(f'{name} must be an ordered pair')
        if block_ratio_range[0] <= 0.0 or block_ratio_range[1] > 1.0:
            raise ValueError('block ratios must lie in (0, 1]')
        if noise_std_range[0] < 0.0 or scale_range[0] <= 0.0:
            raise ValueError('noise std must be non-negative and scale positive')
        self.prob = prob
        self.modes = tuple(modes)
        self.noise_std_range = noise_std_range
        self.block_ratio_range = block_ratio_range
        self.scale_range = scale_range
        self.shift_range = shift_range

    def transform(self, results: dict) -> dict:
        image = results['img']
        if image.ndim != 3 or image.shape[2] < 5:
            raise ValueError(
                'RandomPseudoGeometryCorruption expects BGR + depth + '
                f'reliability channels, got {image.shape}')
        validity = np.ones(image.shape[:2], dtype=np.float32)
        if np.random.random() >= self.prob:
            results['pseudo_corruption'] = 'none'
            results['pseudo_validity'] = validity
            return results

        mode = str(np.random.choice(self.modes))
        depth = image[..., 3].astype(np.float32) / 255.0
        if mode == 'noise':
            sigma = np.random.uniform(*self.noise_std_range)
            depth += np.random.normal(0.0, sigma, depth.shape).astype(np.float32)
            maximum_sigma = max(self.noise_std_range[1], 1e-6)
            validity.fill(np.clip(1.0 - sigma / maximum_sigma, 0.0, 1.0))
        elif mode == 'blur':
            kernel_size = int(np.random.choice((3, 5, 7)))
            depth = cv2.GaussianBlur(depth, (kernel_size, kernel_size), 0)
            validity.fill(1.0 - (kernel_size - 1) / 8.0)
        elif mode == 'block':
            height, width = depth.shape
            ratio_h = np.random.uniform(*self.block_ratio_range)
            ratio_w = np.random.uniform(*self.block_ratio_range)
            block_h = max(1, round(height * ratio_h))
            block_w = max(1, round(width * ratio_w))
            y = np.random.randint(0, max(1, height - block_h + 1))
            x = np.random.randint(0, max(1, width - block_w + 1))
            depth[y:y + block_h, x:x + block_w] = 0.0
            validity[y:y + block_h, x:x + block_w] = 0.0
        elif mode == 'scale_shift':
            scale = np.random.uniform(*self.scale_range)
            shift = np.random.uniform(*self.shift_range)
            depth = scale * depth + shift
            maximum_scale_error = max(
                abs(self.scale_range[0] - 1.0),
                abs(self.scale_range[1] - 1.0),
                1e-6)
            maximum_shift = max(
                abs(self.shift_range[0]), abs(self.shift_range[1]), 1e-6)
            severity = 0.5 * (
                abs(scale - 1.0) / maximum_scale_error
                + abs(shift) / maximum_shift)
            validity.fill(np.clip(1.0 - severity, 0.0, 1.0))
        elif mode == 'zero':
            depth.fill(0.0)
            validity.fill(0.0)

        image = image.copy()
        image[..., 3] = np.clip(depth, 0.0, 1.0) * 255.0
        results['img'] = image
        results['pseudo_corruption'] = mode
        # This target is training-only metadata; it is never appended to the
        # model input.  The reliability head must infer corruption from RGB and
        # geometry rather than reading an explicit flag.
        results['pseudo_validity'] = validity
        return results


@TRANSFORMS.register_module()
class PackRPGVInputs(BaseTransform):
    """Pack standard segmentation data plus the uncropped scene thumbnail."""

    def __init__(self) -> None:
        self.standard_packer = PackSegInputs()

    def transform(self, results: dict) -> dict:
        if 'global_img' not in results:
            raise KeyError(
                'global_img is missing; run GenerateGlobalThumbnail before '
                'PackRPGVInputs')
        packed = self.standard_packer(results)
        thumbnail = np.ascontiguousarray(
            results['global_img'].transpose(2, 0, 1))
        validity = np.asarray(
            results.get(
                'pseudo_validity',
                np.ones(results['img'].shape[:2], dtype=np.float32)),
            dtype=np.float32)
        packed['data_samples'].set_data(dict(
            global_img=PixelData(data=torch.from_numpy(thumbnail)),
            pseudo_validity=PixelData(data=torch.from_numpy(
                np.ascontiguousarray(validity[None])))))
        packed['data_samples'].set_metainfo(dict(
            pseudo_corruption=results.get('pseudo_corruption', 'none')))
        return packed
