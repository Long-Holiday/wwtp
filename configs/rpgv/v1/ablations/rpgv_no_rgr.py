"""RPGV using raw pseudo depth and offline reliability without RGR."""

_base_ = ['../rpgv_stage3_joint.py']

model = dict(component_cfg=dict(
    depth_rectification=False,
    learned_reliability=False,
))
