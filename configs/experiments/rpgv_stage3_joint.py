import os

_base_ = [
    '../_base_/models/rpgv_net.py',
    '../_base_/datasets/wwtp_pseudo_1024x1024.py',
    '../_base_/schedules/rpgv_iter_40k.py',
    '../_base_/default_runtime.py',
]

model = dict(training_stage='joint')
load_from = os.getenv('RPGV_STAGE2_CHECKPOINT') or None
required_previous_stage = 'stage-2 geometry checkpoint'

optim_wrapper = dict(
    accumulative_counts=8,
    paramwise_cfg=dict(
        custom_keys=dict(
            rgb_encoder=dict(lr_mult=0.1),
            global_film=dict(lr_mult=0.25),
            rgb_aux_head=dict(lr_mult=0.25),
            rgb_boundary_head=dict(lr_mult=0.25),
            decoder=dict(lr_mult=0.25),
            refiner=dict(lr_mult=0.25),
            detail_refiner=dict(lr_mult=0.25),
            norm=dict(decay_mult=0.0)),
        norm_decay_mult=0.0))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        save_best='binary/Foreground_IoU',
        rule='greater'))
