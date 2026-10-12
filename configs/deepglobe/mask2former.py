_base_ = [
    '../_base_/models/mask2former_swin_t.py',
    '../_base_/datasets/deepglobe_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
]

model = dict(decode_head=dict(num_classes=2, loss_cls=dict(class_weight=[1.0, 1.0, 0.1])))
default_hooks = dict(checkpoint=dict(save_best='roadIoU', rule='greater'))
work_dir = 'work_dirs/deepglobe/mask2former'
