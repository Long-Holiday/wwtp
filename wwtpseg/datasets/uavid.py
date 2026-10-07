"""UAVid class IDs match GeoSeg's RGB-to-label convention."""
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class UAVidDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('building', 'road', 'tree', 'low_vegetation',
                 'moving_car', 'static_car', 'human', 'clutter'),
        palette=[[128, 0, 0], [128, 64, 128], [0, 128, 0], [128, 128, 0],
                 [64, 0, 128], [192, 0, 192], [64, 64, 0], [0, 0, 0]])

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png',
                         reduce_zero_label=False, **kwargs)
