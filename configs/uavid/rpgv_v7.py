_base_ = ['rpgv_v6.py']

model = dict(
    type='RPGVRemoteSensingV7', contrastive_loss_weight=0.0,
    structure_channels=128, structure_source='depth', structure_feedback=True,
    structure_aux_weight=0.1,
    primitive_cfg=dict(grid_size=8, max_size=64, iterations=2,
                       temperature=0.2, spatial_weight=2.0, fixed_grid=False),
    graph_cfg=dict(heads=4, layers=2, neighbors=8, use_edges=True),
    hierarchical_cfg=dict(node_weight=0.05, relation_weight=0.05,
                          composition_weight=0.05, relation_prediction_weight=0.25,
                          max_relations=128, composition_size=16, negatives=2,
                          margin=0.2, similar_threshold=0.05),
)
work_dir = 'work_dirs/uavid/rpgv_v7'
