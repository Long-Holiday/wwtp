"""Multiclass RPGV adapters for public aerial datasets."""
from .model import RPGVRemoteSensing
from .model_full import RPGVRemoteSensingFull
from .model_v2 import RPGVRemoteSensingV2

__all__ = ['RPGVRemoteSensing', 'RPGVRemoteSensingFull', 'RPGVRemoteSensingV2']
