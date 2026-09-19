optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=10.0, norm_type=2),
    paramwise_cfg=dict(norm_decay_mult=0.0))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=300),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        by_epoch=False,
        begin=300,
        end=3000),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=3000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
auto_scale_lr = dict(enable=False, base_batch_size=16)
