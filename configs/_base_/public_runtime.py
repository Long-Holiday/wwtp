"""Multiclass benchmarks select checkpoints on official validation mIoU."""
_base_ = ['./default_runtime.py']
default_hooks = dict(checkpoint=dict(save_best='mIoU', rule='greater'))
