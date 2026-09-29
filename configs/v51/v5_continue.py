"""Matched continuation: same source, LR, data and budget; original v5 model."""
_base_ = ['./rpgv_v51.py']
model = dict(spatial_refinement=False, boundary_mode='legacy')
work_dir = 'work_dirs/rpgv_v51_v5_continue'
