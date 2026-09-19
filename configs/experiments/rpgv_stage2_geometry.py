import os

_base_ = [
    '../_base_/models/rpgv_net.py',
    '../_base_/datasets/wwtp_pseudo_1024x1024.py',
    '../_base_/schedules/rpgv_iter_20k.py',
    '../_base_/default_runtime.py',
]

model = dict(training_stage='geometry')
load_from = os.getenv('RPGV_STAGE1_CHECKPOINT') or None
required_previous_stage = 'stage-1 RGB checkpoint'

optim_wrapper = dict(
    accumulative_counts=8,
    paramwise_cfg=dict(norm_decay_mult=0.0))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        save_best='binary/Foreground_IoU',
        rule='greater'))
