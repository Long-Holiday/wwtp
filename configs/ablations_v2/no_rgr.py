_base_ = ['../experiments/rpgv_v2_joint.py']
# Preserve raw pseudo-depth and offline Q0, bypassing learned RGR correction
# and reliability attenuation. Geometry encoding/fusion remains active.
model = dict(component_cfg=dict(depth_rectification=False, learned_reliability=False))
