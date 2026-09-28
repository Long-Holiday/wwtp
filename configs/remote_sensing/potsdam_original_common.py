"""Common full-tile Potsdam protocol: 22 train, 2_10 val, 14 official test."""

import os

_base_ = ['./_base_/runtime.py']

data_root = os.getenv('POTSDAM_DATA_ROOT', 'data/remote_sensing/prepared/potsdam')
# The complete 300 m x 300 m scene is resized for a single L4 GPU. Thus the
# evaluation grid is 1536 px (0.1953125 m/px), not the original 5 cm grid.
input_size = 1536
custom_imports = dict(imports=['wwtpseg.remote_sensing'], allow_failed_imports=False)

train_pipeline = [
    dict(type='LoadPotsdamRGB', size=input_size),
    dict(type='LoadPotsdamColorAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=180, pad_val=0, seg_pad_val=255),
    dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadPotsdamRGB', size=input_size),
    dict(type='LoadPotsdamColorAnnotations'),
    dict(type='SetPotsdamEvaluationShape'),
    dict(type='PackSegInputs'),
]

def dataset(split, pipeline):
    return dict(type='PotsdamOriginalDataset', data_root=data_root,
                split=split, pipeline=pipeline)

train_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dataset('train', train_pipeline),
)
val_dataloader = dict(
    batch_size=1, num_workers=1, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dataset('val', test_pipeline),
)
test_dataloader = dict(
    batch_size=1, num_workers=1, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dataset('test', test_pipeline),
)
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
    dict(type='PotsdamFiveClassMetric', erode_radius=0, prefix='full'),
]

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-4, by_epoch=False, begin=0, end=100),
    dict(type='PolyLR', eta_min=0, power=1.0, by_epoch=False, begin=100, end=3000),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=3000, val_interval=250)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=250,
                    max_keep_ckpts=3, save_best='mIoU', rule='greater'),
)
