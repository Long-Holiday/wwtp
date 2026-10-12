# Shared training protocol, expressed in optimizer updates: an effective batch of
# 8 samples, 7,500 updates and 60,000 image exposures, bfloat16 autocast, and one
# validation and checkpoint every 250 updates.
#
# A model whose micro batch has to be smaller keeps the same contract by raising
# accumulative_counts and scaling max_iters and every schedule boundary by that
# same factor, so the LR at each update, the image budget and the validation
# cadence stay identical across models. The protocol test enforces those
# invariants per dataset.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    # bfloat16 spans fp32's exponent range and needs no loss scaling;
    # `tools/train.py --amp-dtype bfloat16` writes the same value.
    loss_scale=1.0,
    accumulative_counts=1,
    optimizer=dict(
        type='AdamW', lr=6e-4, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.01),
    clip_grad=dict(max_norm=10.0, norm_type=2),
    paramwise_cfg=dict(norm_decay_mult=0.0))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=250),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        by_epoch=False,
        begin=250,
        end=7500),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=7500, val_interval=250)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
auto_scale_lr = dict(enable=False, base_batch_size=8)

enable_early_stopping = False
