"""RPGV v1/v2 geometry stages retain synchronized corruption supervision."""
_base_ = ['./wwtp_rgb_1024x1024.py']
import os

pseudo_root = os.getenv('WWTP_PSEUDO_ROOT', os.path.join(_base_.data_root, 'pseudo_geometry'))
train_pipeline = [step for step in _base_.train_pipeline
                  if step['type'] not in ('RandomFlip', 'PackRPGVInputs')] + [
    dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
    dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomPseudoGeometryCorruption', prob=0.3),
    dict(type='PackRPGVInputs')]
test_pipeline = [step for step in _base_.test_pipeline
                 if step['type'] != 'PackRPGVInputs'] + [
    dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
    dict(type='PackRPGVInputs')]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
