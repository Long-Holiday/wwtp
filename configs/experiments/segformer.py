_base_ = [
    '../_base_/models/segformer_mit_b2.py',
    '../_base_/datasets/wwtp_512x512.py',
    '../_base_/schedules/iter_40k.py',
    '../_base_/default_runtime.py',
]

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0)),
        norm_decay_mult=0.0))
