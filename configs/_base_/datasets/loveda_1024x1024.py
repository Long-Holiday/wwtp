_base_ = []
import os

benchmark_dataset = 'loveda'
data_root = os.getenv('LOVEDA_DATA_ROOT', 'data/remote_sensing/prepared/loveda')
train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    dict(type='RequireImageShape', size=(1024, 1024)),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    dict(type='RequireImageShape', size=(1024, 1024)), dict(type='PackSegInputs'),
]
predict_pipeline = [dict(type='LoadImageFromFile'),
                    dict(type='RequireImageShape', size=(1024, 1024)), dict(type='PackSegInputs')]
train_dataloader = dict(
    batch_size=8, num_workers=4, persistent_workers=True, pin_memory=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type='LoveDADataset', data_root=data_root,
                 data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8, num_workers=2, persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='LoveDADataset', data_root=data_root,
                 data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
                 pipeline=test_pipeline))
# Official test labels are private. Local scores use only the official validation split.
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator
predict_dataset = dict(type='LoveDADataset', data_root=data_root,
                       data_prefix=dict(img_path='img_dir/test'),
                       pipeline=predict_pipeline)
