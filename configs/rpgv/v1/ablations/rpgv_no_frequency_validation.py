"""RPGV with direct scale-matched geometry residuals instead of DFGV."""

_base_ = ['../rpgv_stage3_joint.py']

model = dict(component_cfg=dict(frequency_validation=False))
