_base_ = [
    '../_base_/datasets/uavid_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
]

model = dict(
    type='HDNetBaseline', base_channel=48, num_classes=8,
    data_preprocessor=dict(
        type='SegDataPreProcessor', mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375], bgr_to_rgb=True,
        pad_val=0, seg_pad_val=255, size=(1024, 1024),
        test_cfg=dict(size_divisor=32)),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

work_dir = 'work_dirs/uavid/hdnet'

# Measured on the target RTX 3090 with the real training entry (2026-10-12):
# batch 8 and batch 4 exceed 24 GiB; batch 2 peaks at 20235 MiB.
# The protocol is expressed in optimizer updates, so this model keeps it with a
# smaller micro batch and a proportionally larger accumulation count: the effective
# batch stays 8 samples, the LR at each of the 7,500 updates, the 60,000-image budget
# and the validation/checkpoint cadence match every other model.
train_dataloader = dict(batch_size=2)
optim_wrapper = dict(accumulative_counts=4, dtype='float16', loss_scale='dynamic')
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(type='PolyLR', eta_min=0.0, power=0.9, by_epoch=False, begin=1000, end=30000),
]
train_cfg = dict(max_iters=30000, val_interval=1000)
default_hooks = dict(logger=dict(interval=100), checkpoint=dict(interval=1000))
protocol_amp_override = (
    "torch's grid_sampler_2d_cuda has no bfloat16 kernel, so HDNet can only run mixed "
    "precision under fp16; the evaluation batch drops to 2 to keep one validation pass inside the same budget")
