"""RPGV without the stride-4 high-frequency boundary residual."""

_base_ = ['../experiments/rpgv_stage3_joint.py']

model = dict(component_cfg=dict(boundary_fusion=False))
