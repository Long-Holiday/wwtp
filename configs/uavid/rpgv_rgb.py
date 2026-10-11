_base_ = [
    '../_base_/models/rpgv_net.py',
    '../_base_/datasets/uavid_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
]
import os

custom_imports = dict(imports=['wwtpseg', 'wwtpseg.models.segmentors.multiclass', 'mmdet.models'], allow_failed_imports=False)
model = dict(
    _delete_=True, type='RPGVRemoteSensing', num_classes=8,
    rgb_encoder={{_base_.model.rgb_encoder}},
    data_preprocessor={{_base_.model.data_preprocessor}},
    decoder_channels=64, detail_channels=24, use_global_context=False,
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
work_dir = 'work_dirs/uavid/rpgv_rgb'
