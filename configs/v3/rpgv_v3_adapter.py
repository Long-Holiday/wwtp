import os

_base_ = ['../experiments/rpgv_v2_stage3_joint.py']

model = dict(
    type='RPGVNetV3', training_stage='joint', adapter_channels=24,
    rgb_encoder=dict(init_cfg=None),
    max_geometry_correction=3.0, uncertainty_floor=0.2,
    correction_strength=1.0, use_geometry_evidence=True,
    error_loss_weight=0.25, protection_loss_weight=0.5,
    adapter_boundary_weight=0.1, adapter_region_weight=0.05,
    geometry_dropout_prob=0.1)
load_from = os.getenv('RPGV_V3_INIT_CHECKPOINT') or None
required_previous_stage = 'verified v3 initialization (tools/initialize_rpgv_v3.py)'
work_dir = 'work_dirs/rpgv_v3_adapter'
enable_early_stopping = False
custom_hooks = []
optim_wrapper = dict(
    _delete_=True, type='AmpOptimWrapper', loss_scale='dynamic', accumulative_counts=8,
    optimizer=dict(type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=5.0, norm_type=2),
    paramwise_cfg=dict(norm_decay_mult=0.0))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', eta_min=1e-6, power=0.9, by_epoch=False, begin=500, end=12000)]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=12000, val_interval=1000)
default_hooks = dict(checkpoint=dict(interval=1000, max_keep_ckpts=3))
