"""Potsdam RPGV with dataset-supplied normalized DSM in place of Depth Anything.

Architecture, training schedule, augmentation and test split inherit the
Depth Anything experiment unchanged. Only the geometry file root differs.
"""

from copy import deepcopy
import os

_base_ = ['./potsdam_rpgv_full_geometry.py']

dsm_root = os.path.join(_base_.data_root, 'dsm_geometry')

train_pipeline = deepcopy(_base_.train_pipeline)
test_pipeline = deepcopy(_base_.test_pipeline)
for pipeline in (train_pipeline, test_pipeline):
    geometry_loaders = [step for step in pipeline if step['type'] == 'LoadPseudoGeometry']
    if len(geometry_loaders) != 1:
        raise ValueError('Expected exactly one geometry loader in each pipeline')
    geometry_loaders[0]['pseudo_root'] = dsm_root

train_dataloader = deepcopy(_base_.train_dataloader)
for dataset in train_dataloader['dataset']['datasets']:
    dataset['pipeline'] = train_pipeline

test_dataloader = deepcopy(_base_.test_dataloader)
test_dataloader['dataset']['pipeline'] = test_pipeline
