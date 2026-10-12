# Downgraded micro batch for models whose activations do not fit the reference
# batch of 8 on a 24 GiB card. The protocol is expressed in optimizer updates, so
# halving the micro batch and doubling both the accumulation count and every
# iteration boundary keeps the effective batch at 8 samples, the LR at each of the
# 7,500 updates, the 60,000-image budget and the validation/checkpoint cadence
# identical to every other model. Include this file last in `_base_` and record the
# measured peak memory of the model it is applied to in the leaf config.
train_dataloader = dict(batch_size=4)
optim_wrapper = dict(accumulative_counts=2)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', eta_min=0.0, power=0.9, by_epoch=False, begin=500, end=15000),
]
train_cfg = dict(max_iters=15000, val_interval=500)
default_hooks = dict(logger=dict(interval=50), checkpoint=dict(interval=500))
