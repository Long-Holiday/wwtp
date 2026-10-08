_base_ = [
    '../_base_/models/hrnet_w18.py',
    '../_base_/datasets/wwtp_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/default_runtime.py',
]

train_dataloader = dict(batch_size=2)

work_dir = 'work_dirs/wwtp/hrnet'
