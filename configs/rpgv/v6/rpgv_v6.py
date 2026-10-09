_base_ = ['../v5/rpgv_v5.py']
"""Optimized V6: decoder geometry and residual contours, 5,000 joint updates."""

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
    contrastive_temperature=0.07,
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
    dict(type='LoadPseudoGeometry', pseudo_root={{_base_.pseudo_root}}, required=True),
    dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackRPGVContourInputs', contour_truncation=5.0),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={
        'rgb_encoder': dict(lr_mult=0.1),
        'logit_scale': dict(decay_mult=0.0),
        'branch_gains': dict(decay_mult=0.0),
    }))
load_from = None
resume = False
work_dir = 'work_dirs/rpgv_v6_optimized_1m'
