_base_ = [
    '../_base_/models/mask2former_swin_t.py',
    '../_base_/datasets/deepglobe_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
]

model = dict(decode_head=dict(num_classes=2, loss_cls=dict(class_weight=[1.0, 1.0, 0.1])))
default_hooks = dict(checkpoint=dict(save_best='roadIoU', rule='greater'))
work_dir = 'work_dirs/deepglobe/mask2former'

# Measured on the target RTX 3090 with the real training entry (2026-10-12):
# batch 8 exceeds 24 GiB even in mixed precision; batch 4 peaks at 15078 MiB.
# The protocol is expressed in optimizer updates, so this model keeps it with a
# smaller micro batch and a proportionally larger accumulation count: the effective
# batch stays 8 samples, the LR at each of the 7,500 updates, the 60,000-image budget
# and the validation/checkpoint cadence match every other model.
train_dataloader = dict(batch_size=4)
optim_wrapper = dict(accumulative_counts=2, dtype='float16', loss_scale='dynamic')
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', eta_min=0.0, power=0.9, by_epoch=False, begin=500, end=15000),
]
train_cfg = dict(max_iters=15000, val_interval=500)
default_hooks = dict(logger=dict(interval=50), checkpoint=dict(interval=500, save_best='roadIoU', rule='greater'))
protocol_amp_override = (
    "mmcv's ms_deform_attn_forward_cuda has no bfloat16 kernel, so the deformable attention "
    "this model is built on can only run under fp16 autocast")
