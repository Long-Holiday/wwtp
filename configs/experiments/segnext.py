"""SegNeXt-MSCAN-S: full-scene 1024 / 1m training, 5,000 optimizer updates.

Protocol matches RPGV v5 full-scene 1m training:
- Input resolution: 1024x1024 full scene (1m GSD), whole image evaluation
- Data root: wwtp_semantic_dataset_1m (bypassing WWTP_DATA_ROOT which docker-compose points to old 0.5m data)
- Batch size: 2 x 4 accumulation = effective batch size 8
- Updates: 5,000 updates = 20,000 micro-iterations = 40,000 image exposures
- Metrics: 1m physical metrics (boundary tolerance 1.5px=1.5m, topology 64m^2)
- Work dir: work_dirs/segnext_1m (protecting historical work_dirs/segnext baseline)
"""
import os

_base_ = [
    '../_base_/models/segnext_mscan_s.py',
    '../_base_/default_runtime.py',
]

# Deliberately do not inherit WWTP_DATA_ROOT: docker-compose sets the old root.
data_root = os.getenv('WWTP_SEGNEXT_DATA_ROOT',
                      os.getenv('WWTP_V5_DATA_ROOT', 'wwtp_semantic_dataset_1m'))

# Model adaptation for 1024 full scene and whole-image evaluation
model = dict(
    data_preprocessor=dict(size=(1024, 1024)),
    test_cfg=dict(_delete_=True, mode='whole'))

# Full scenes preserve 1m GSD and the original positive/negative sample ratio.
# Diagonal/horizontal/vertical flips are exact pixel-grid operations.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.75,
         direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=2, num_workers=4, persistent_workers=True, pin_memory=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type='WWTPDataset', data_root=data_root,
                 data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='WWTPDataset', data_root=data_root,
                 data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
                 pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='WWTPDataset', data_root=data_root,
                 data_prefix=dict(img_path='images/test', seg_map_path='annotations/test'),
                 pipeline=test_pipeline))

# Preserve the old physical BF tolerance (3 old pixels = 1.5m).
# This does not eliminate the rasterization difference between resolutions.
val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'], nan_to_num=0),
    dict(type='BinaryBoundaryMetric', boundary_tolerance=1.5, pixel_size_m=1.0),
    dict(type='BinaryTopologyMetric', small_area=64),  # 64m², formerly 256px
]
test_evaluator = val_evaluator

# SegNeXt's official recipe uses full FP32 precision (OptimWrapper instead of
# AmpOptimWrapper) because the LightHamHead hamburger matrix decomposition is
# prone to underflow/overflow in FP16. It also uses lr=6e-5 with a 10x head multiplier.
optim_wrapper = dict(
    type='OptimWrapper',
    accumulative_counts=4,
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0)),
        norm_decay_mult=0.0))

# MMEngine steps these schedulers per dataloader iteration, not optimizer step.
# 2 images × 4 accumulation × 5,000 updates = 40,000 image exposures.
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type='PolyLR', eta_min=0.0, power=0.9, by_epoch=False, begin=1000, end=20000),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=3,
        save_best='binary/Foreground_IoU',
        rule='greater'))
enable_early_stopping = False
randomness = dict(seed=42, deterministic=False)
load_from = None
resume = False
work_dir = 'work_dirs/segnext_1m'
