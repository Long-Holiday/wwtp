"""V5: full-scene 1024 / 1m training, 5,000 optimizer updates."""
import os

_base_ = [
    '../_base_/models/rpgv_net.py',
    '../_base_/schedules/rpgv_iter_40k.py',
    '../_base_/default_runtime.py',
]

# Deliberately do not inherit WWTP_DATA_ROOT: docker-compose sets the old root.
data_root = os.getenv('WWTP_V5_DATA_ROOT', 'wwtp_semantic_dataset_1m')
pseudo_root = os.path.join(data_root, 'pseudo_geometry')
model = dict(
    _delete_=True, type='RPGVNetV5',
    rgb_encoder={{_base_.model.rgb_encoder}},
    data_preprocessor={{_base_.model.data_preprocessor}},
    rgb_channels=[64, 128, 320, 512], decoder_channels=64,
    geometry_channels=[16, 32], detail_channels=24,
    use_geometry=True, geometry_input='depth', use_context=True,
    geometry_dropout_prob=0.1, use_contour=True,
    contour_truncation=5.0, max_logit_correction=2.0,
    coarse_loss_weight=0.3, region_loss_weight=0.05,
    final_boundary_loss_weight=0.1, sdf_loss_weight=0.1,
    test_cfg=dict(mode='whole'))

# Full scenes preserve 1m GSD and the original positive/negative sample ratio.
# Diagonal/horizontal/vertical flips are exact pixel-grid operations.
train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    dict(type='PhotoMetricDistortion'),
    dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
    dict(type='RandomFlip', prob=0.75,
         direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
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

optim_wrapper = dict(
    accumulative_counts=4,
    loss_scale=dict(_delete_=True, init_scale=128.0),
    paramwise_cfg=dict(custom_keys=dict(rgb_encoder=dict(lr_mult=0.1)),
                       norm_decay_mult=0.0))
# MMEngine steps these schedulers per dataloader iteration, not optimizer step.
# 2 images × 4 accumulation × 5,000 updates = 40,000 image exposures.
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type='PolyLR', eta_min=0.0, power=0.9, by_epoch=False, begin=1000, end=20000),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
default_hooks = dict(checkpoint=dict(interval=1000))
enable_early_stopping = False
randomness = dict(seed=42, deterministic=False)
load_from = None
resume = False
work_dir = 'work_dirs/rpgv_v5_1m'
