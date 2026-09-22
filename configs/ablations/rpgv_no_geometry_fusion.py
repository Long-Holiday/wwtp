"""RPGV without either geometry residual injection scale."""

_base_ = ['../experiments/rpgv_stage3_joint.py']

model = dict(component_cfg=dict(
    boundary_fusion=False,
    region_fusion=False,
))
