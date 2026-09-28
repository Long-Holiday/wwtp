_base_ = ['../experiments/rpgv_v2_joint.py']
# Keep SDF supervision; isolate its direct correction of segmentation logits.
model = dict(use_contour=False)
