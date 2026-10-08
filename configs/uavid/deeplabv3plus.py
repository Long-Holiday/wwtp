_base_ = [
    '../_base_/models/deeplabv3plus_r50.py',
    '../_base_/datasets/uavid_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
]

model = dict(decode_head=dict(num_classes=8), auxiliary_head=dict(num_classes=8), test_cfg=dict(_delete_=True, mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)},
        norm_decay_mult=0.0))

work_dir = 'work_dirs/uavid/deeplabv3plus'
