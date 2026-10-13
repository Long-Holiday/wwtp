_base_ = ['./rpgv_v81.py']
# Remove only the Gaussian kernel from assignment; coverage/shape/edges remain.
model = dict(gaussian_cfg=dict(spatial_support=False))
work_dir = 'work_dirs/wwtp/rpgv_v81_content_only_support'
