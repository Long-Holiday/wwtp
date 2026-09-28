_base_ = ['../experiments/rpgv_v2_joint.py']
# Remove the joint final-mask region/boundary constraint as ONE strategy.
# SDF, coarse segmentation and the RGB boundary auxiliary stay supervised.
model = dict(region_loss_weight=0.0, final_boundary_loss_weight=0.0)
