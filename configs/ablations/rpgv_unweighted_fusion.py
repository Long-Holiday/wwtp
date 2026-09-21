"""RPGV injecting bounded residuals without final reliability weighting."""

_base_ = ['../experiments/rpgv_stage3_joint.py']

model = dict(component_cfg=dict(reliability_weighting=False))
