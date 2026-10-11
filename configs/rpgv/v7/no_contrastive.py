_base_ = ['rpgv_v7.py']
# Keep every inference module and structure segmentation supervision.
model = dict(hierarchical_cfg=dict(node_weight=0.0, relation_weight=0.0, composition_weight=0.0))
work_dir = 'work_dirs/rpgv_v7_no_contrastive_1m'
