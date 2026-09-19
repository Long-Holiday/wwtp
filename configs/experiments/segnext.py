_base_ = [
    '../_base_/models/segnext_mscan_s.py',
    '../_base_/datasets/wwtp_512x512.py',
    '../_base_/schedules/iter_40k.py',
    '../_base_/default_runtime.py',
]

# SegNeXt's official recipe uses full FP32 precision (OptimWrapper instead of
# AmpOptimWrapper) because the LightHamHead hamburger matrix decomposition is
# prone to underflow/overflow in FP16. It also uses lr=6e-5 with a 10x head multiplier.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0)),
        norm_decay_mult=0.0))
