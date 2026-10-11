_base_ = [
    '../_base_/models/mask2former_swin_t.py',
    '../_base_/datasets/uavid_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
]

model = dict(decode_head=dict(num_classes=8, loss_cls=dict(class_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1])), test_cfg=dict(_delete_=True, mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

work_dir = 'work_dirs/uavid/mask2former'
