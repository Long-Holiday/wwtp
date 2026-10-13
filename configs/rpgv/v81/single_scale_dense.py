_base_ = ['./rpgv_v81.py']
# 400 single-scale anchors vs 336 multiscale anchors; not equal compute/capacity.
model = dict(gaussian_cfg=dict(grid_sizes=(20,)))
work_dir = 'work_dirs/wwtp/rpgv_v81_single_scale_dense'
