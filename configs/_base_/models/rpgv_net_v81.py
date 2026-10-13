_base_ = ['./rpgv_net_v8.py']

# V8.1 retains V8's RGB-only backbone, decoder and dense BCE/node objectives.
model = dict(
    type='RPGVNetV81',
    gaussian_cfg=dict(grid_sizes=(4, 8, 16), max_size=64,
                      content_aware=True, content_scale=2.0),
    graph_cfg=dict(use_edge_value=True),
    local_detail=True,
    region_loss_weight=0.05,
    final_boundary_loss_weight=0.1,
    balanced_boundary=True,
    gaussian_dense_dice_weight=0.05)
