_base_ = [
    '../_base_/models/unetformer_r18.py',
    '../_base_/datasets/loveda_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
]

model = dict(decode_head=dict(num_classes=7), auxiliary_head=dict(num_classes=7))

work_dir = 'work_dirs/loveda/unetformer'
