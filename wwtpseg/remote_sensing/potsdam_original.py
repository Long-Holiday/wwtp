"""Read the full-resolution Potsdam RGB PNGs and original color TIFF labels."""

from pathlib import Path

import numpy as np
from PIL import Image
from mmcv.transforms import BaseTransform
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS


PARTICIPANT_TILES = set('''2_10 2_11 2_12 3_10 3_11 3_12 4_10 4_11 4_12
5_10 5_11 5_12 6_10 6_11 6_12 6_7 6_8 6_9 7_10 7_11 7_12 7_7 7_8 7_9'''.split())
TEST_TILES = set('''2_13 2_14 3_13 3_14 4_13 4_14 4_15 5_13 5_14
5_15 6_13 6_14 6_15 7_13'''.split())
VAL_TILES = {'2_10'}
EXCLUDED_TILES = {'7_10'}
SPLIT_TILES = {
    'train': PARTICIPANT_TILES - VAL_TILES - EXCLUDED_TILES,
    'val': VAL_TILES,
    'test': TEST_TILES,
}


@DATASETS.register_module()
class PotsdamOriginalDataset(BaseSegDataset):
    """Fixed 22/1/14 tile split without materializing image or label patches."""

    METAINFO = dict(
        classes=('impervious_surface', 'building', 'low_vegetation', 'tree',
                 'car', 'clutter'),
        palette=[[255, 255, 255], [0, 0, 255], [0, 255, 255],
                 [0, 255, 0], [255, 255, 0], [255, 0, 0]],
    )

    def __init__(self, split: str, geometry_root: str | None = None,
                 test_no_boundary: bool = False, **kwargs):
        if split not in SPLIT_TILES:
            raise ValueError(f'Unknown Potsdam split: {split}')
        if test_no_boundary and split != 'test':
            raise ValueError('The noBoundary reference is only used for test')
        self.split = split
        self.geometry_root = Path(geometry_root) if geometry_root else None
        self.test_no_boundary = test_no_boundary
        super().__init__(img_suffix='.png', seg_map_suffix='.tif',
                         reduce_zero_label=False, **kwargs)

    def load_data_list(self) -> list[dict]:
        root = Path(self.data_root)
        image_dir = root / '2_Ortho_RGB' / '2_Ortho_RGB'
        if self.split == 'test':
            label_dir = root / ('5_Labels_all_noBoundary' if self.test_no_boundary
                                else '5_Labels_all')
        else:
            label_dir = root / '5_Labels_for_participants' / '5_Labels_for_participants'
        data = []
        for tile in sorted(SPLIT_TILES[self.split]):
            stem = f'top_potsdam_{tile}'
            image = image_dir / f'{stem}_RGB.png'
            label_suffix = '_label_noBoundary.tif' if self.test_no_boundary else '_label.tif'
            label = label_dir / f'{stem}{label_suffix}'
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f'Potsdam {self.split}/{tile}: {image} or {label}')
            item = dict(img_path=str(image), seg_map_path=str(label),
                        label_map=self.label_map, reduce_zero_label=False,
                        seg_fields=[], tile_id=tile)
            if self.geometry_root:
                geometry = self.geometry_root / f'{stem}_RGB.npz'
                if not geometry.is_file():
                    raise FileNotFoundError(geometry)
                item['pseudo_geometry_path'] = str(geometry)
            data.append(item)
        return data


@TRANSFORMS.register_module()
class LoadPotsdamRGB(BaseTransform):
    """Decode a whole RGB tile at the experiment's full-scene working grid."""

    def __init__(self, size: int = 1536):
        self.size = size

    def transform(self, results: dict) -> dict:
        with Image.open(results['img_path']) as source:
            if source.size != (6000, 6000):
                raise ValueError(f'Unexpected Potsdam RGB size: {source.size}')
            resized = source.convert('RGB').resize(
                (self.size, self.size), Image.Resampling.LANCZOS)
            rgb = np.asarray(resized)
        # Existing MMSeg preprocessors and RPGVRemoteSensingFull expect BGR.
        results['img'] = np.ascontiguousarray(rgb[:, :, ::-1])
        results['img_shape'] = (self.size, self.size)
        results['ori_shape'] = (self.size, self.size)
        results['source_shape'] = (6000, 6000)
        return results


@TRANSFORMS.register_module()
class LoadPotsdamColorAnnotations(BaseTransform):
    """Convert official RGB label colors to the six MMSeg class IDs in memory."""

    COLORS = {
        (255, 255, 255): 0, (0, 0, 255): 1,
        (0, 255, 255): 2, (0, 255, 0): 3,
        (255, 255, 0): 4, (255, 0, 0): 5,
        (0, 0, 0): 255,
    }

    def __init__(self, size: int = 1536):
        self.size = size

    def transform(self, results: dict) -> dict:
        with Image.open(results['seg_map_path']) as image:
            if image.size != (6000, 6000):
                raise ValueError(f'Unexpected Potsdam label size: {image.size}')
            rgb = np.asarray(image.convert('RGB').resize(
                (self.size, self.size), Image.Resampling.NEAREST))
        packed = (rgb[:, :, 0].astype(np.uint32) << 16
                  | rgb[:, :, 1].astype(np.uint32) << 8
                  | rgb[:, :, 2].astype(np.uint32))
        labels = np.full(packed.shape, 255, dtype=np.uint8)
        known = np.zeros(packed.shape, dtype=bool)
        for (r, g, b), class_id in self.COLORS.items():
            mask = packed == ((r << 16) | (g << 8) | b)
            labels[mask] = class_id
            known |= mask
        if not known.all():
            unknown = np.unique(packed[~known])[:8]
            raise ValueError(f'Unexpected Potsdam label colors in '
                             f'{results["seg_map_path"]}: {unknown.tolist()}')
        results['gt_seg_map'] = labels
        results['seg_fields'].append('gt_seg_map')
        return results


@TRANSFORMS.register_module()
class SetPotsdamEvaluationShape(BaseTransform):
    """Evaluate predictions and labels on the common resized full-tile grid."""

    def transform(self, results: dict) -> dict:
        results.setdefault('source_shape', results['ori_shape'])
        results['ori_shape'] = results['img'].shape[:2]
        return results
