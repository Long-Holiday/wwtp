_base_ = ['rpgv_v7.py']
# Keep all structure objectives; remove only F16/F8 structure injection.
model = dict(structure_feedback=False)
work_dir = 'work_dirs/rpgv_v7_no_feedback_1m'
