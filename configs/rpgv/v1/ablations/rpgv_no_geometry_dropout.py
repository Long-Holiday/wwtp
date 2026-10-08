"""RPGV joint training without per-sample geometry dropout."""

_base_ = ['../rpgv_stage3_joint.py']

model = dict(geometry_dropout_prob=0.0)
