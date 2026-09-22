"""CBR-Net comparison on the unchanged WWTP RGB data and metrics."""

_base_ = [
    '../_base_/datasets/wwtp_512x512.py',
    '../_base_/schedules/iter_40k.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['wwtpseg', 'wwtpseg.edge_baselines', 'mmdet.models'],
    allow_failed_imports=False)

model = dict(
    type='CBRNetBaseline',
    pretrained=True,
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512),
        test_cfg=dict(size_divisor=32)),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))

train_dataloader = dict(batch_size=1)
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=6e-4,
                                    betas=(0.9, 0.999), weight_decay=0.01))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        by_epoch=False,
        begin=500,
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
        min_delta=0.01,
        patience=3,
        strict=False),
]

work_dir = 'work_dirs/edge_baselines/cbr_net'
