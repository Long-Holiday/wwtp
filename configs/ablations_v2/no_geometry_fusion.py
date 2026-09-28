_base_ = ['../experiments/rpgv_v2_joint.py']
# Bypass both residual injections; geometry auxiliary training remains present.
model = dict(component_cfg=dict(boundary_fusion=False, region_fusion=False))
