_base_ = [
    '../_base_/models/rpgv_net.py',
    '../_base_/datasets/wwtp_pseudo_1024x1024.py',
    '../_base_/schedules/rpgv_iter_40k.py',
    '../_base_/default_runtime.py',
]

# Reuse the v2 ImageNet MiT architecture, never a trained stage checkpoint.
model = dict(
    _delete_=True, type='RPGVNetV4',
    rgb_encoder={{_base_.model.rgb_encoder}},
    data_preprocessor={{_base_.model.data_preprocessor}},
    rgb_channels=[64, 128, 320, 512], geometry_channels=[16, 32, 64, 128],
    decoder_channels=64, detail_channels=24,
    use_geometry=True, use_frequency=True, context_grid=4,
    use_global_context=False, geometry_dropout_prob=0.1,
    use_contour=True, contour_truncation=10.0, max_logit_correction=2.0,
    coarse_loss_weight=0.3, region_loss_weight=0.1,
    final_boundary_loss_weight=0.1, sdf_loss_weight=0.1,
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# No reliability-prediction target, synthetic corruption pass, or thumbnail
# encoding. Keep v2's synchronized five-channel spatial augmentations.
train_pipeline = [
    step for step in _base_.train_pipeline
    if step['type'] not in ('GenerateGlobalThumbnail', 'RandomPseudoGeometryCorruption', 'PackRPGVInputs')
] + [dict(type='PackSegInputs')]
test_pipeline = [
    step for step in _base_.test_pipeline
    if step['type'] not in ('GenerateGlobalThumbnail', 'PackRPGVInputs')
] + [dict(type='PackSegInputs')]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

optim_wrapper = dict(
    accumulative_counts=8,
    paramwise_cfg=dict(custom_keys=dict(rgb_encoder=dict(lr_mult=0.1)),
                       norm_decay_mult=0.0))
default_hooks = dict(checkpoint=dict(interval=2000))
enable_early_stopping = False
load_from = None
work_dir = 'work_dirs/rpgv_v4'
