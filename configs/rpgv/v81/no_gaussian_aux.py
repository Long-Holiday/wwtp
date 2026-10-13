_base_ = ['./rpgv_v81.py']
# Disable the entire Gaussian auxiliary objective group, including V8.1 dense Dice.
model = dict(gaussian_dense_weight=0.0, gaussian_node_weight=0.0,
             gaussian_dense_dice_weight=0.0)
work_dir = 'work_dirs/wwtp/rpgv_v81_no_gaussian_aux'
