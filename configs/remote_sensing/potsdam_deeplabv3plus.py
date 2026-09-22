import os

_base_ = ['../_base_/models/deeplabv3plus_r50.py', './_base_/runtime.py']
data_root = os.getenv('POTSDAM_DATA_ROOT', 'data/remote_sensing/prepared/potsdam')

model = dict(
    decode_head=dict(
        num_classes=6,
        loss_decode=[dict(type='CrossEntropyLoss', loss_weight=1.0, avg_non_ignore=True)],
    ),
    auxiliary_head=dict(
        num_classes=6,
        loss_decode=dict(type='CrossEntropyLoss', loss_weight=0.4, avg_non_ignore=True),
    ),
    test_cfg=dict(mode='whole'),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=180, pad_val=0, seg_pad_val=255),
    dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='PotsdamDataset',
                data_root=data_root,
                data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
                pipeline=train_pipeline,
            ),
            dict(
                type='PotsdamDataset',
                data_root=data_root,
                data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
                pipeline=train_pipeline,
            ),
        ],
    ),
)
val_dataloader = None
val_evaluator = None
val_cfg = None

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PotsdamDataset',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
        pipeline=test_pipeline,
    ),
)

test_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
    dict(type='PotsdamFiveClassMetric', erode_radius=0, prefix='full'),
    dict(type='PotsdamFiveClassMetric', erode_radius=3, prefix='eroded3'),
]

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=1500, end=32000),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=32000)
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        max_keep_ckpts=3,
    )
)
