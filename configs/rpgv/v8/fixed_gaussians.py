_base_ = ['./rpgv_v8.py']
# Fix centers/covariances; amplitude, height, content and relations still learn.
model = dict(gaussian_cfg=dict(fixed_gaussians=True))
work_dir = 'work_dirs/wwtp/rpgv_v8_fixed_gaussians'
