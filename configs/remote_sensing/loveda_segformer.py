import os

_base_ = ['../_base_/models/segformer_mit_b2.py', './_base_/runtime.py']
data_root = os.getenv('LOVEDA_DATA_ROOT', 'data/remote_sensing/prepared/loveda')
crop_size = (512, 512)

model = dict(
    decode_head=dict(
        num_classes=7,
        loss_decode=[dict(type='CrossEntropyLoss', loss_weight=1.0, avg_non_ignore=True)]))

train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'), dict(type='PackSegInputs'),
]
test_pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'), dict(type='PackSegInputs')]

train_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type='LoveDADataset', data_root=data_root,
                 data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='LoveDADataset', data_root=data_root,
                 data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
                 pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator
