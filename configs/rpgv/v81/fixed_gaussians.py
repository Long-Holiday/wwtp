_base_ = ['./rpgv_v81.py']
# Fix centers/covariances; content, amplitude, latent height and relations still learn.
model = dict(gaussian_cfg=dict(fixed_gaussians=True))
work_dir = 'work_dirs/wwtp/rpgv_v81_fixed_gaussians'
