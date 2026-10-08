"""RPGV using uncorrected pseudo depth while retaining reliability learning."""

_base_ = ['../rpgv_stage3_joint.py']

model = dict(component_cfg=dict(depth_rectification=False))
