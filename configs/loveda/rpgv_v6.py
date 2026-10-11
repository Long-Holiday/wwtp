_base_ = [
    '../_base_/models/rpgv_net.py',
    '../_base_/datasets/loveda_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
]
import os

custom_imports = dict(
    imports=['wwtpseg', 'wwtpseg.models.segmentors.multiclass', 'mmdet.models'],
    allow_failed_imports=False)
_rgb_encoder = dict(_base_.model.rgb_encoder)
_rgb_encoder.update(type='MixVisionTransformerSDPA', with_cp=True)
model = dict(
    _delete_=True, type='RPGVRemoteSensingV6', num_classes=7,
    rgb_encoder=_rgb_encoder,
    data_preprocessor={{_base_.model.data_preprocessor}},
    rgb_channels=[64, 128, 320, 512], decoder_channels=64, detail_channels=24,
    geometry_channels=[32, 64], use_geometry=True, geometry_input='depth',
    use_context=True, use_contour=True, geometry_dropout_prob=0.0,
    fusion_location='decoder', contour_mode='residual', rectifier_channels=16,
    max_offset=2.0, max_depth_correction=0.1, frequency_cutoff=0.15,
    frequency_padding=8,
    cluster_cfg=dict(
        grid_size=4, max_cluster_size=64, iterations=2,
        temperature=0.2, spatial_weight=2.0,
        min_reliability=0.35, min_region_mass=2.0, min_region_energy=1e-4),
    attention_cfg=dict(num_heads=4, query_size=32, kv_size=8),
    fusion_scale=0.1, contrastive_embed_dim=64, contrastive_temperature=0.1,
    contrastive_logit_bias=-10.0, contrastive_loss_weight=0.05,
    rectification_loss_weight=0.01, spatial_refinement=True,
    contour_truncation=5.0, coarse_loss_weight=0.3, sdf_loss_weight=0.1,
    boundary_mode='band', boundary_band_loss_weight=0.05, boundary_band_radius=3,
    test_cfg=dict(mode='whole'))
del _rgb_encoder
work_dir = 'work_dirs/loveda/rpgv_v6'
pseudo_root = os.getenv('PUBLIC_PSEUDO_ROOT', os.path.join(_base_.data_root, 'pseudo_geometry'))
train_pipeline = [
    step for step in _base_.train_pipeline if step['type'] not in ('RandomFlip', 'PackSegInputs')
] + [dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
     dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
     dict(type='PackSegInputs')]
test_pipeline = [step for step in _base_.test_pipeline
                 if step['type'] != 'PackSegInputs'] + [
    dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
    dict(type='PackSegInputs')]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
predict_pipeline = [step for step in _base_.predict_pipeline
                    if step['type'] != 'PackSegInputs'] + [
    dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
    dict(type='PackSegInputs')]
predict_dataset = dict(pipeline=predict_pipeline)
