"""Categorical public-dataset adapter for the independent RGB-only v8."""
from mmseg.registry import MODELS

from ..rpgv_net_v8 import RPGVNetV8


@MODELS.register_module()
class RPGVRemoteSensingV8(RPGVNetV8):
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, binary=False, **kwargs)
