"""Same six-class RPGV v2 network with official Potsdam nDSM geometry."""

import os

_base_ = ['./potsdam_original_rpgv_v2_depth_anything.py']

geometry_root = os.getenv(
    'POTSDAM_NDSM_ROOT',
    os.path.join(_base_.data_root, 'geometry_official_ndsm_1536'))

train_pipeline = [dict(step) for step in _base_.train_pipeline]
test_pipeline = [dict(step) for step in _base_.test_pipeline]
for pipeline in (train_pipeline, test_pipeline):
    next(step for step in pipeline if step['type'] == 'LoadPseudoGeometry')[
        'pseudo_root'] = geometry_root

train_dataloader = dict(dataset=dict(
    geometry_root=geometry_root, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(
    geometry_root=geometry_root, pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(
    geometry_root=geometry_root, pipeline=test_pipeline))
