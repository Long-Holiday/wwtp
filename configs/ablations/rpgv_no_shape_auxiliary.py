"""RPGV without explicit boundary and signed-distance auxiliary losses."""

_base_ = ['../experiments/rpgv_stage3_joint.py']

model = dict(loss_weights=dict(boundary=0.0, sdf=0.0))
