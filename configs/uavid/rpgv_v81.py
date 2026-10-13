_base_ = [
    '../_base_/models/rpgv_net_v81.py',
    '../_base_/datasets/uavid_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
]

custom_imports = dict(imports=['wwtpseg', 'wwtpseg.models.segmentors.multiclass', 'mmdet.models'],
                      allow_failed_imports=False)
model = dict(type='RPGVRemoteSensingV81', num_classes=8,
             test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
work_dir = 'work_dirs/uavid/rpgv_v81'
