"""Categorical public-dataset adapter for RGB-only RPGV v8.1."""
from mmseg.registry import MODELS

from ..rpgv_net_v81 import RPGVNetV81


@MODELS.register_module()
class RPGVRemoteSensingV81(RPGVNetV81):
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, binary=False, **kwargs)
