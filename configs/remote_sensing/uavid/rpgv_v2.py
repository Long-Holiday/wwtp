_base_ = [
    '../../_base_/models/rpgv_net.py', '../../_base_/datasets/uavid_1024x1024.py',
    '../../_base_/schedules/baseline_20k.py', '../../_base_/public_runtime.py',
]
import os

custom_imports = dict(imports=['wwtpseg', 'wwtpseg.remote_sensing', 'mmdet.models'], allow_failed_imports=False)
model = dict(
    _delete_=True, type='RPGVRemoteSensingV2', num_classes=8,
    rgb_encoder={{_base_.model.rgb_encoder}},
    data_preprocessor={{_base_.model.data_preprocessor}},
    decoder_channels=64, detail_channels=24, use_global_context=False,
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
optim_wrapper = dict(paramwise_cfg=dict(custom_keys=dict(rgb_encoder=dict(lr_mult=0.1))))
work_dir = 'work_dirs/uavid/rpgv_v2'
pseudo_root = os.getenv('PUBLIC_PSEUDO_ROOT', os.path.join(_base_.data_root, 'pseudo_geometry'))
train_pipeline = [
    step for step in _base_.train_pipeline if step['type'] not in ('RandomFlip', 'PackSegInputs')
] + [dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
     dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
     dict(type='PackSegInputs')]
test_pipeline = [step for step in _base_.test_pipeline
                 if step['type'] != 'PackSegInputs'] + [
    dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
    dict(type='PackSegInputs')]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

predict_pipeline = [step for step in _base_.predict_pipeline
                    if step['type'] != 'PackSegInputs'] + [
    dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
    dict(type='PackSegInputs')]
