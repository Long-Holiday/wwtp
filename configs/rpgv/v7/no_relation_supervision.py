_base_ = ['rpgv_v7.py']
# Remove relation contrast and measurement prediction, but retain composition ranking.
model = dict(hierarchical_cfg=dict(relation_weight=0.0))
work_dir = 'work_dirs/rpgv_v7_no_relation_supervision_1m'
