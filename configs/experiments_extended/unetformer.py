_base_ = ['../experiments/unetformer.py']

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=3000,
        end=3500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        by_epoch=False,
        begin=3500,
        end=20000),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=3,
        save_best='binary/Foreground_IoU',
        rule='greater'))

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='binary/Foreground_IoU',
        rule='greater',
        min_delta=1.0,
        patience=3,
        strict=False),
]

