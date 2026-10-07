_base_ = [
    '../../_base_/models/rpgv_net.py', '../../_base_/datasets/loveda_1024x1024.py',
    '../../_base_/schedules/baseline_20k.py', '../../_base_/public_runtime.py',
]
import os

custom_imports = dict(imports=['wwtpseg', 'wwtpseg.remote_sensing', 'mmdet.models'], allow_failed_imports=False)
model = dict(
    _delete_=True, type='RPGVRemoteSensing', num_classes=7,
    rgb_encoder={{_base_.model.rgb_encoder}},
    data_preprocessor={{_base_.model.data_preprocessor}},
    decoder_channels=64, detail_channels=24, use_global_context=False,
    test_cfg=dict(mode='whole'))
optim_wrapper = dict(paramwise_cfg=dict(custom_keys=dict(rgb_encoder=dict(lr_mult=0.1))))
work_dir = 'work_dirs/loveda/rpgv_rgb'
