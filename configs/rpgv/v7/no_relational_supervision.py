_base_ = ['rpgv_v7.py']
# Keep the graph, node alignment, structure auxiliary head and decoder feedback.
model = dict(hierarchical_cfg=dict(relation_weight=0.0, composition_weight=0.0))
work_dir = 'work_dirs/rpgv_v7_no_relational_supervision_1m'
