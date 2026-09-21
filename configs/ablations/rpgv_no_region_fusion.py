"""RPGV without the stride-16 semantic region residual."""

_base_ = ['../experiments/rpgv_stage3_joint.py']

model = dict(component_cfg=dict(region_fusion=False))
