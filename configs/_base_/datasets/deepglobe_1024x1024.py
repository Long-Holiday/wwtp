_base_ = []
import os

benchmark_dataset = 'deepglobe'
data_root = os.getenv('DEEPGLOBE_DATA_ROOT', 'data/remote_sensing/prepared/deepglobe')
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
    batch_size=2, num_workers=4, persistent_workers=True, pin_memory=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type='DeepGlobeRoadDataset', data_root=data_root,
                 data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='DeepGlobeRoadDataset', data_root=data_root,
                 data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
                 pipeline=test_pipeline))
# These are labeled local holdouts from official train, not the private challenge test.
test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='DeepGlobeRoadDataset', data_root=data_root,
                 data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
                 pipeline=test_pipeline))
val_evaluator = dict(type='RoadIoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator
predict_dataset = dict(type='DeepGlobeRoadDataset', data_root=data_root,
                       data_prefix=dict(img_path='img_dir/official_test'),
                       pipeline=predict_pipeline)
