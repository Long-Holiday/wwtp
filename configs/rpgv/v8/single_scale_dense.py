_base_ = ['./rpgv_v8.py']
# Single-scale control with MORE anchors than the multi-scale default (400 vs 336),
# so a win for the default cannot be explained by node count alone.
model = dict(gaussian_cfg=dict(grid_sizes=(20,)))
work_dir = 'work_dirs/wwtp/rpgv_v8_single_scale_dense'
