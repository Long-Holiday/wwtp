_base_ = ['./progressive_base.py']
model = dict(component_cfg=dict(depth_rectification=True, learned_reliability=True))
