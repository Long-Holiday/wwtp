"""Load offline pseudo depth and reliability maps for RPGV-Net."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadPseudoGeometry(BaseTransform):
    """Append pseudo depth and reliability to an RGB image.

    The offline generator stores one ``.npz`` file per source image with
    quantized ``depth`` and ``reliability`` arrays.  Floating-point archives
    in [0, 1] are accepted as well.  Both maps are appended to
    ``results['img']`` as 0--255 float channels.  Keeping all five channels in
    one array lets MMSeg apply exactly the same resize, rotate, crop and flip
    to RGB and geometry without maintaining a second transform pipeline.

    Args:
        pseudo_root: Directory containing ``<split>/<stem>.npz`` files.  The
            split is inferred from the RGB image's parent directory.
        depth_key: Array name for normalized pseudo depth.
        reliability_key: Array name for the consistency reliability map.
        suffix: Geometry archive suffix.
        required: Raise when an archive is missing.  Disabling this is useful
            only for debugging and appends neutral depth/reliability maps.
    """

    def __init__(
        self,
        pseudo_root: str,
        depth_key: str = 'depth',
        reliability_key: str = 'reliability',
        suffix: str = '.npz',
        required: bool = True,
    ) -> None:
        self.pseudo_root = Path(pseudo_root).expanduser()
        self.depth_key = depth_key
        self.reliability_key = reliability_key
        self.suffix = suffix
        self.required = required

    def _geometry_path(self, results: dict) -> Path:
        if results.get('pseudo_geometry_path'):
            return Path(results['pseudo_geometry_path'])
        image_path = Path(results['img_path'])
        split = image_path.parent.name
        return self.pseudo_root / split / f'{image_path.stem}{self.suffix}'

    @staticmethod
    def _validate_map(
        value: np.ndarray, name: str, shape: tuple[int, int]
    ) -> np.ndarray:
        value = np.asarray(value).squeeze()
        if value.shape != shape:
            raise ValueError(
                f'{name} has shape {value.shape}, expected image shape {shape}')
        if np.issubdtype(value.dtype, np.integer):
            value = value.astype(np.float32) / np.iinfo(value.dtype).max
        else:
            value = value.astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f'{name} contains NaN or infinite values')
        return np.clip(value, 0.0, 1.0)

    def transform(self, results: dict) -> dict:
        image = results['img']
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                'LoadPseudoGeometry must run after RGB-only photometric '
                f'augmentation and before geometry transforms; got {image.shape}')

        path = self._geometry_path(results)
        height, width = image.shape[:2]
        if not path.is_file():
            if self.required:
                raise FileNotFoundError(
                    f'pseudo geometry not found: {path}. Run '
                    '`python tools/generate_pseudo_geometry.py` first.')
            depth = np.full((height, width), 0.5, dtype=np.float32)
            reliability = np.zeros((height, width), dtype=np.float32)
        else:
            with np.load(path) as archive:
                depth = self._validate_map(
                    archive[self.depth_key], self.depth_key, (height, width))
                reliability = self._validate_map(
                    archive[self.reliability_key], self.reliability_key,
                    (height, width))

        geometry = np.stack([depth, reliability], axis=-1) * 255.0
        results['img'] = np.concatenate(
            [image.astype(np.float32, copy=False), geometry], axis=-1)
        results['pseudo_geometry_path'] = str(path)
        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'pseudo_root={str(self.pseudo_root)!r}, required={self.required})')
