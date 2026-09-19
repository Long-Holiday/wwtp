_base_ = [
    '../_base_/models/rs_mamba_tiny.py',
    '../_base_/datasets/wwtp_512x512.py',
    '../_base_/schedules/iter_40k.py',
    '../_base_/default_runtime.py',
]

# The eight-direction scan is memory intensive. Accumulation keeps the same
# effective batch size as the other baselines on common 24 GB GPUs.
train_dataloader = dict(batch_size=1)

# Use robust FP32 optimization with gradient clipping to prevent recurrent SSM gradient overflow
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    accumulative_counts=4,
    optimizer=dict(
        type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        custom_keys=dict(
            a_logs=dict(decay_mult=0.0),
            ds=dict(decay_mult=0.0),
        )),
)
