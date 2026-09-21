"""RPGV using the decoder coarse logit without stride-4 residual refinement."""

_base_ = ['../experiments/rpgv_stage3_joint.py']

model = dict(component_cfg=dict(boundary_refinement=False))
