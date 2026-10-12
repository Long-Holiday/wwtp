_base_ = ['rpgv_v7.py']
# Remove all hierarchical objectives, including relation prediction; keep structure segmentation.
model = dict(hierarchical_cfg=dict(node_weight=0.0, relation_weight=0.0, composition_weight=0.0))
work_dir = 'work_dirs/rpgv_v7_no_hierarchical_supervision_1m'
