_base_ = ['./rpgv_v81.py']
# Whole Gaussian branch removal, not a parameter-matched RGB baseline.
model = dict(use_gaussians=False)
work_dir = 'work_dirs/wwtp/rpgv_v81_rgb_only'
