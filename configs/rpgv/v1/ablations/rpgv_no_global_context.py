"""RPGV without the shared full-scene token and FiLM modulation."""

_base_ = ['../rpgv_stage3_joint.py']

model = dict(component_cfg=dict(global_context=False))
