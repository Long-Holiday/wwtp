"""MMSegmentation dataset definition for the WWTP binary dataset."""

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class WWTPDataset(BaseSegDataset):
    """Wastewater-treatment-plant semantic-segmentation dataset.

    Mask values are kept unchanged: 0 is background, 1 is WWTP and 255 is
    reserved as the ignore index. World files next to PNGs are deliberately
    ignored by the ``img_suffix``/``seg_map_suffix`` filtering.
    """

    METAINFO = dict(
        classes=('background', 'wastewater_plant'),
        palette=[[0, 0, 0], [0, 255, 0]],
    )

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs,
        )
