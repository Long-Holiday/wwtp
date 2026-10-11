"""Use v7 inference/structure losses with v6's multiclass semantic objectives."""
from mmseg.registry import MODELS

from ..rpgv_net_v7 import RPGVNetV7
from .model_v6 import RPGVRemoteSensingV6


@MODELS.register_module()
class RPGVRemoteSensingV7(RPGVNetV7, RPGVRemoteSensingV6):
    """Cooperative initialization creates class-specific contour and structure heads."""
