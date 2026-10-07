_base_ = [
    '../../_base_/models/mask2former_swin_t.py',
    '../../_base_/datasets/loveda_1024x1024.py',
    '../../_base_/schedules/baseline_20k.py', '../../_base_/public_runtime.py',
]

model = dict(decode_head=dict(num_classes=7, loss_cls=dict(class_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1])))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    accumulative_counts=4,
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
            'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'relative_position_bias_table': dict(
                lr_mult=0.1, decay_mult=0.0),
        },
        norm_decay_mult=0.0))

work_dir = 'work_dirs/loveda/mask2former'
