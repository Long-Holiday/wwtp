_base_ = ['./rpgv_v8.py']
# Freeze the learned center reach at one lattice pitch per level, i.e. the bound
# that the multi-scale default replaces with a learned factor.
model = dict(gaussian_cfg=dict(center_scale_learnable=False))
work_dir = 'work_dirs/wwtp/rpgv_v8_bounded_centers'
