_base_ = [
    '../../_base_/models/segnext_mscan_s.py',
    '../../_base_/datasets/uavid_1024x1024.py',
    '../../_base_/schedules/baseline_20k.py', '../../_base_/public_runtime.py',
]

model = dict(decode_head=dict(num_classes=8), test_cfg=dict(_delete_=True, mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

optim_wrapper = dict(
    _delete_=True, type='OptimWrapper',
    accumulative_counts=4,
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0)),
        norm_decay_mult=0.0))

work_dir = 'work_dirs/uavid/segnext'
