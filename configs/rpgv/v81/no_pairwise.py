_base_ = ['./rpgv_v81.py']
model = dict(region_loss_weight=0.0, final_boundary_loss_weight=0.0)
work_dir = 'work_dirs/wwtp/rpgv_v81_no_pairwise'
