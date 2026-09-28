_base_ = ['../experiments/rpgv_v2_joint.py']
# Keep the contour correction; learn its field through segmentation losses.
model = dict(loss_weights=dict(sdf=0.0))
