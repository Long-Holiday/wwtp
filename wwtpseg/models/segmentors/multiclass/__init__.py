"""Multiclass RPGV adapters for public aerial datasets."""
from .model import RPGVRemoteSensing
from .model_full import RPGVRemoteSensingFull
from .model_v2 import RPGVRemoteSensingV2
from .model_v6 import RPGVRemoteSensingV6
from .model_v7 import RPGVRemoteSensingV7

__all__ = ['RPGVRemoteSensing', 'RPGVRemoteSensingFull', 'RPGVRemoteSensingV2',
           'RPGVRemoteSensingV6', 'RPGVRemoteSensingV7']
