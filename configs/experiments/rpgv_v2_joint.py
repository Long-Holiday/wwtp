"""Single-stage v2: every active branch trains from the first iteration."""
_base_ = [
    '../_base_/models/rpgv_net.py',
    '../_base_/datasets/wwtp_pseudo_1024x1024.py',
    '../_base_/schedules/rpgv_iter_40k.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='RPGVNetV2', training_stage='joint',
    decoder_channels=64, detail_channels=24,
    contour_truncation=10.0, max_logit_correction=2.0,
    coarse_loss_weight=0.3, region_loss_weight=0.1,
    correction_smoothness_weight=0.02, use_contour=True,
    final_boundary_loss_weight=0.1, learnable_loss_weights=False,
    # Keep final segmentation dominant and balance RGB/geometry auxiliaries.
    # Boundary=0.2 yields an effective RGB-boundary coefficient of 0.1 in v2.
    loss_weights=dict(final=1.0, rgb=0.2, geometry=0.2, boundary=0.2,
                      sdf=0.1, reliability=0.05, preserve=0.05, equivariance=0.05))

# ImageNet initialization belongs to rgb_encoder.init_cfg. No segmentation
# checkpoint, staged warm start, or stage-dependent loss/LR multipliers.
load_from = None
resume = False
# One explicit common rule for Full and every ablation, including resumes.
enable_early_stopping = True
custom_imports = dict(imports=['wwtpseg', 'mmdet.models', 'wwtpseg.engine'],
                      allow_failed_imports=False)
custom_hooks = [dict(type='RPGVEarlyStoppingHook', monitor='binary/Foreground_IoU',
                     rule='greater', min_delta=0.1, patience=5, strict=True)]
randomness = dict(seed=42, deterministic=True)
env_cfg = dict(cudnn_benchmark=False)
optim_wrapper = dict(
    accumulative_counts=8,
    paramwise_cfg=dict(
        custom_keys=dict(rgb_encoder=dict(lr_mult=0.1),
                         norm=dict(decay_mult=0.0)),
        norm_decay_mult=0.0))

# Match the former 40k + 20k + 40k iteration budget, with one continuous
# schedule. Joint iterations cost more than RGB-only/geometry-only iterations.
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=0.9, by_epoch=False,
         begin=1500, end=100000),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=2000)
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', by_epoch=False, interval=2000, max_keep_ckpts=3,
    save_best='binary/Foreground_IoU', rule='greater'))
