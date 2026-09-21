"""RPGV using the offline reliability prior without task calibration."""

_base_ = ['../experiments/rpgv_stage3_joint.py']

model = dict(component_cfg=dict(learned_reliability=False))
