"""V2 architecture under the v5 full-scene data/update protocol."""
_base_ = ['../rpgv_v5.py']
model = dict(
    _delete_=True, type='RPGVNetV2',
    rgb_encoder={{_base_.model.rgb_encoder}},
    data_preprocessor={{_base_.model.data_preprocessor}},
    rgb_channels=[64, 128, 320, 512], geometry_channels=[32, 64, 128, 256],
    decoder_channels=64, detail_channels=24,
    training_stage='joint', use_global_context=False,
    geometry_dropout_prob=0.1, use_contour=True,
    contour_truncation=5.0, max_logit_correction=2.0,
    coarse_loss_weight=0.3, region_loss_weight=0.05,
    final_boundary_loss_weight=0.1, learnable_loss_weights=False,
    loss_weights=dict(final=1.0, rgb=0.2, geometry=0.2, boundary=0.2,
                      sdf=0.1, reliability=0.05, preserve=0.05, equivariance=0.05),
    test_cfg=dict(mode='whole'))
work_dir = 'work_dirs/rpgv_v5_control_v2_1m'
