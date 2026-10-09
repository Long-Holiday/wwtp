_base_ = ['rpgv_v6.py']

model = dict(contrastive_loss_weight=0.0)
work_dir = 'work_dirs/rpgv_v6_optimized_no_contrastive_1m'
