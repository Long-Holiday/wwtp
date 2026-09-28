"""V4 architecture under exactly the v5 full-scene data/update protocol."""
_base_ = ['../rpgv_v5.py']
model = dict(
    _delete_=True, type='RPGVNetV4',
    rgb_encoder={{_base_.model.rgb_encoder}},
    data_preprocessor={{_base_.model.data_preprocessor}},
    rgb_channels=[64, 128, 320, 512], geometry_channels=[16, 32, 64, 128],
    decoder_channels=64, detail_channels=24,
    use_geometry=True, use_frequency=True, context_grid=4,
    use_global_context=False, geometry_dropout_prob=0.1,
    use_contour=True, contour_truncation=5.0, max_logit_correction=2.0,
    coarse_loss_weight=0.3, region_loss_weight=0.05,
    final_boundary_loss_weight=0.1, sdf_loss_weight=0.1,
    test_cfg=dict(mode='whole'))
work_dir = 'work_dirs/rpgv_v5_control_v4_1m'
