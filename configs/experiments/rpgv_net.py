_base_ = [
    '../_base_/models/rpgv_net.py',
    '../_base_/datasets/wwtp_pseudo_1024x1024.py',
    '../_base_/schedules/iter_40k.py',
    '../_base_/default_runtime.py',
]

# A 1024 px crop is intentionally batch-size one; accumulation preserves the
# same effective batch size as the lighter baselines.
optim_wrapper = dict(
    accumulative_counts=4,
    paramwise_cfg=dict(
        custom_keys=dict(
            rgb_encoder=dict(lr_mult=0.1),
            norm=dict(decay_mult=0.0)),
        norm_decay_mult=0.0))
