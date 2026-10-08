"""RPGV using the offline reliability prior without task calibration."""

_base_ = ['../rpgv_stage3_joint.py']

model = dict(component_cfg=dict(learned_reliability=False))
