_base_ = ['./rpgv_v5.py']
# This is the Full vs progressive_contour distinction in the old quick screen.
model = dict(region_loss_weight=0.0, final_boundary_loss_weight=0.0)
work_dir = 'work_dirs/rpgv_v5_1m_no_structure_losses'
