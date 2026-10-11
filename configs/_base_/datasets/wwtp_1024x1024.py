_base_ = []
import os

crop_size = (1024, 1024)
data_root = os.getenv('WWTP_DATA_ROOT', os.getenv('WWTP_V5_DATA_ROOT', 'wwtp_semantic_dataset_1m'))
train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    dict(type='RequireImageShape', size=crop_size),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    dict(type='RequireImageShape', size=crop_size), dict(type='PackSegInputs'),
]
train_num_workers = int(os.getenv('WWTP_NUM_WORKERS', '8'))
eval_num_workers = int(os.getenv('WWTP_VAL_NUM_WORKERS', '4'))

train_dataloader = dict(
    batch_size=2, num_workers=train_num_workers, persistent_workers=True, pin_memory=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type='WWTPDataset', data_root=data_root,
                 data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=eval_num_workers, persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='WWTPDataset', data_root=data_root,
                 data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
                 pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1, num_workers=eval_num_workers, persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='WWTPDataset', data_root=data_root,
                 data_prefix=dict(img_path='images/test', seg_map_path='annotations/test'),
                 pipeline=test_pipeline))
val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'], nan_to_num=0),
    dict(type='BinaryBoundaryMetric', boundary_tolerance=1.5, pixel_size_m=1.0),
    dict(type='BinaryTopologyMetric', small_area=64),
]
test_evaluator = val_evaluator
