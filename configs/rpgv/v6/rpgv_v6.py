_base_ = ['../v5/rpgv_v5.py']
"""Optimized V6: decoder geometry and residual contours, 5,000 joint updates."""
import os
import mmengine


data_root = os.getenv('WWTP_DATA_ROOT', os.getenv('WWTP_V5_DATA_ROOT', 'wwtp_semantic_dataset_1m'))
pseudo_root = os.getenv('WWTP_PSEUDO_ROOT', os.path.join(data_root, 'pseudo_geometry'))


model = dict(
    type='RPGVNetV6',
    rgb_encoder=dict(type='MixVisionTransformerSDPA', with_cp=True),
    geometry_channels=[32, 64],
    geometry_dropout_prob=0.0,
    fusion_location='decoder',
    contour_mode='residual',
    rectifier_channels=16,
    max_offset=2.0,
    max_depth_correction=0.1,
    frequency_cutoff=0.15,
    frequency_padding=8,
    cluster_cfg=dict(
        grid_size=4, max_cluster_size=64, iterations=2,
        temperature=0.2, spatial_weight=2.0,
        min_reliability=0.35, min_region_mass=2.0, min_region_energy=1e-4),
    attention_cfg=dict(num_heads=4, query_size=32, kv_size=8),
    fusion_scale=0.1,
    contrastive_embed_dim=64,
    contrastive_temperature=0.1,
    contrastive_logit_bias=-10.0,
    contrastive_loss_weight=0.05,
    rectification_loss_weight=0.01,
    spatial_refinement=True,
    boundary_mode='band',
    boundary_band_loss_weight=0.05,
    boundary_band_radius=3,
)

train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    dict(type='RequireImageShape', size=(1024, 1024)),
    dict(type='PhotoMetricDistortion'),
    dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
    dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackRPGVContourInputs', contour_truncation=5.0),
]
train_dataloader = dict(batch_size=8, dataset=dict(data_root=data_root, pipeline=train_pipeline))
test_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    dict(type='RequireImageShape', size=(1024, 1024)),
    dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
    dict(type='PackSegInputs'),
]
val_dataloader = dict(batch_size=8, dataset=dict(data_root=data_root, pipeline=test_pipeline))
test_dataloader = dict(batch_size=8, dataset=dict(data_root=data_root, pipeline=test_pipeline))

_protocol = mmengine.Config.fromfile(os.path.join(
    '{{ fileDirname }}', '../../_base_/schedules/baseline_20k.py'))
optim_wrapper = dict(_delete_=True, **_protocol.optim_wrapper)
param_scheduler = _protocol.param_scheduler
train_cfg = dict(_delete_=True, **_protocol.train_cfg)
auto_scale_lr = _protocol.auto_scale_lr
enable_early_stopping = _protocol.enable_early_stopping
del _protocol
# v5 pins a 1000-iteration checkpoint interval, which would survive the shared
# runtime change; the protocol fixes it in updates like the schedule does.
default_hooks = dict(logger=dict(interval=25), checkpoint=dict(interval=250))
load_from = None
resume = False
work_dir = 'work_dirs/rpgv_v6_siglip_1m'
