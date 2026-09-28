_base_ = ['../experiments/rpgv_v2_joint.py']
# Preserve the RGB boundary auxiliary coefficient (0.1).
model = dict(final_boundary_loss_weight=0.0)
