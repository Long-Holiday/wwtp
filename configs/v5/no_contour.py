_base_ = ['./rpgv_v5.py']
# Keep auxiliary SDF supervision; isolate its application to final logits.
model = dict(use_contour=False)
work_dir = 'work_dirs/rpgv_v5_1m_no_contour'
