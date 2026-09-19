_base_ = [
    '../_base_/models/unetformer_r18.py',
    '../_base_/datasets/wwtp_512x512.py',
    '../_base_/schedules/iter_40k.py',
    '../_base_/default_runtime.py',
]

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'backbone.encoder': dict(lr_mult=0.1)},
        norm_decay_mult=0.0))
