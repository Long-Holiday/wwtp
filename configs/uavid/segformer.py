_base_ = [
    '../_base_/models/segformer_mit_b2.py',
    '../_base_/datasets/uavid_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
]

model = dict(decode_head=dict(num_classes=8), test_cfg=dict(_delete_=True, mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

work_dir = 'work_dirs/uavid/segformer'

# Measured on the target RTX 3090 (2026-10-12, real training entry, bfloat16):
# batch 8 needs more than 24 GiB and batch 4 peaks at 18.9 GiB steady,
# leaving no room for a validation pass.
# The protocol is expressed in optimizer updates, so this model keeps it with a
# smaller micro batch and a proportionally larger accumulation count: the effective
# batch stays 8 samples, the LR at each of the 7,500 updates, the 60,000-image
# budget and the validation/checkpoint cadence match every other model.
train_dataloader = dict(batch_size=2)
optim_wrapper = dict(accumulative_counts=4)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(type='PolyLR', eta_min=0.0, power=0.9, by_epoch=False, begin=1000, end=30000),
]
train_cfg = dict(max_iters=30000, val_interval=1000)
default_hooks = dict(logger=dict(interval=100), checkpoint=dict(interval=1000))
