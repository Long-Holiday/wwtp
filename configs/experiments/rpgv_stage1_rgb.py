_base_ = [
    '../_base_/models/rpgv_net.py',
    '../_base_/datasets/wwtp_rgb_1024x1024.py',
    '../_base_/schedules/rpgv_iter_40k.py',
    '../_base_/default_runtime.py',
]

model = dict(training_stage='rgb')

optim_wrapper = dict(
    accumulative_counts=8,
    paramwise_cfg=dict(
        custom_keys=dict(
            rgb_encoder=dict(lr_mult=0.1),
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
