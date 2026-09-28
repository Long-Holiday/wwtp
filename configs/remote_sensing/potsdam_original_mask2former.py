_base_ = ['./potsdam_original_common.py', '../_base_/models/mask2former_swin_t.py']

custom_imports = dict(imports=['mmdet.models', 'wwtpseg.remote_sensing'],
                      allow_failed_imports=False)
model = dict(
    backbone=dict(with_cp=True),
    decode_head=dict(num_classes=6, loss_cls=dict(
        type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0,
        reduction='mean', class_weight=[1.0] * 6 + [0.1])),
    test_cfg=dict(mode='whole'),
)
optim_wrapper = dict(
    _delete_=True, type='AmpOptimWrapper', loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05, eps=1e-8,
                   betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
        'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
        'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
        'relative_position_bias_table': dict(lr_mult=0.1, decay_mult=0.0),
    }, norm_decay_mult=0.0),
)
