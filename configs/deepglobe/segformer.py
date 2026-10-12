_base_ = [
    '../_base_/models/segformer_mit_b2.py',
    '../_base_/datasets/deepglobe_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
    '../_base_/schedules/micro_batch_2.py',
]

model = dict(decode_head=dict(num_classes=2))
default_hooks = dict(checkpoint=dict(save_best='roadIoU', rule='greater'))
work_dir = 'work_dirs/deepglobe/segformer'
