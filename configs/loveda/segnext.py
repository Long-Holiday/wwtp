_base_ = [
    '../_base_/models/segnext_mscan_s.py',
    '../_base_/datasets/loveda_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
]

model = dict(decode_head=dict(num_classes=7))

work_dir = 'work_dirs/loveda/segnext'
