_base_ = ['./rpgv_v8.py']
# Fix centers/covariances; amplitude, height, content and relations still learn.
# Each anchor keeps its own level's pitch (covariance 0.8 * step of that level);
# do not collapse the per-level step back into one shared value.
model = dict(gaussian_cfg=dict(fixed_gaussians=True))
work_dir = 'work_dirs/wwtp/rpgv_v8_fixed_gaussians'
