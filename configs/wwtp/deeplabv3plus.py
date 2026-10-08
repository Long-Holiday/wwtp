_base_ = [
    '../_base_/models/deeplabv3plus_r50.py',
    '../_base_/datasets/wwtp_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/default_runtime.py',
]

train_dataloader = dict(batch_size=2)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)},
        norm_decay_mult=0.0))


work_dir = 'work_dirs/wwtp/deeplabv3plus'
