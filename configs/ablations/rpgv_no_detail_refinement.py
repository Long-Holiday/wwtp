"""RPGV without the stride-2 RGB detail refinement stage."""

_base_ = ['../experiments/rpgv_stage3_joint.py']

model = dict(component_cfg=dict(detail_refinement=False))
