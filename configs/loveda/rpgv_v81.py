_base_ = [
    '../_base_/models/rpgv_net_v81.py',
    '../_base_/datasets/loveda_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
]

custom_imports = dict(imports=['wwtpseg', 'wwtpseg.models.segmentors.multiclass', 'mmdet.models'],
                      allow_failed_imports=False)
model = dict(type='RPGVRemoteSensingV81', num_classes=7)
work_dir = 'work_dirs/loveda/rpgv_v81'
