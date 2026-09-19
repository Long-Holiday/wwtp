import os

dataset_type = 'WWTPDataset'
data_root = os.getenv('WWTP_DATA_ROOT', 'wwtp_semantic_dataset')
crop_size = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateGlobalThumbnail', size=(512, 512)),
    dict(
        type='RandomResize',
        scale=(2048, 2048),
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(
        type='RandomRotate',
        prob=0.5,
        degree=180,
        pad_val=0,
        seg_pad_val=255),
    dict(
        type='RandomForegroundCrop',
        crop_size=crop_size,
        foreground_prob=0.8,
        foreground_label=1,
        edge_margin_ratio=0.15,
        min_foreground_pixels=64),
    dict(
        type='RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackRPGVInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='GenerateGlobalThumbnail', size=(512, 512)),
    dict(type='PackRPGVInputs'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations/test'),
        pipeline=test_pipeline))

metrics = [
    dict(
        type='IoUMetric',
        iou_metrics=['mIoU', 'mDice', 'mFscore'],
        nan_to_num=0),
    dict(
        type='BinaryBoundaryMetric',
        boundary_tolerance=3,
        pixel_size_m=0.5),
]
val_evaluator = metrics
test_evaluator = metrics
