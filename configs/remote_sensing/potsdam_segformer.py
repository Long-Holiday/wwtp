import os

_base_ = ['../_base_/models/segformer_mit_b2.py', './_base_/runtime.py']
data_root = os.getenv('POTSDAM_DATA_ROOT', 'data/remote_sensing/prepared/potsdam')

model = dict(
    decode_head=dict(
        num_classes=6,
        loss_decode=[dict(type='CrossEntropyLoss', loss_weight=1.0, avg_non_ignore=True)]),
    test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=180, pad_val=0, seg_pad_val=255),
    dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PhotoMetricDistortion'), dict(type='PackSegInputs'),
]
test_pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'), dict(type='PackSegInputs')]

def split_dataset(split):
    return dict(type='PotsdamDataset', data_root=data_root,
                data_prefix=dict(img_path=f'img_dir/{split}', seg_map_path=f'ann_dir/{split}'),
                pipeline=train_pipeline if split == 'train' else test_pipeline)

train_dataloader = dict(batch_size=4, num_workers=4, persistent_workers=True,
                        sampler=dict(type='InfiniteSampler', shuffle=True),
                        dataset=split_dataset('train'))
val_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=split_dataset('val'))
test_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True,
                       sampler=dict(type='DefaultSampler', shuffle=False),
                       dataset=split_dataset('test'))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
    dict(type='PotsdamFiveClassMetric', erode_radius=0, prefix='full'),
    dict(type='PotsdamFiveClassMetric', erode_radius=3, prefix='eroded3'),
]

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=1500, end=32000),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=32000, val_interval=4000)
