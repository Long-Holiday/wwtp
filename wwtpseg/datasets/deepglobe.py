"""DeepGlobe masks are converted to background=0/road=1 during preparation."""
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class DeepGlobeRoadDataset(BaseSegDataset):
    METAINFO = dict(classes=('background', 'road'), palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png',
                         reduce_zero_label=False, **kwargs)
