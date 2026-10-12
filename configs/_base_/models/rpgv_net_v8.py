checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'

# V8 receives raw BGR and normalizes once inside the model; no geometry channels.
model = dict(
    type='RPGVNetV8',
    data_preprocessor=dict(type='SegDataPreProcessor', bgr_to_rgb=False,
                           pad_val=0, seg_pad_val=255, size=(1024, 1024),
                           test_cfg=dict(size_divisor=32)),
    rgb_encoder=dict(
        type='MixVisionTransformerSDPA', in_channels=3, embed_dims=64,
        num_stages=4, num_layers=[3, 4, 6, 3], num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3], sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, with_cp=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    rgb_channels=[64, 128, 320, 512], decoder_channels=64, detail_channels=24,
    gaussian_channels=128,
    gaussian_cfg=dict(grid_sizes=(4, 8, 16), max_size=64, fixed_gaussians=False),
    graph_cfg=dict(heads=4, layers=2, use_contact=True), use_gaussians=True, use_message_passing=True,
    use_context=True, use_contour=True, spatial_refinement=True,
    coarse_loss_weight=0.3, sdf_loss_weight=0.1, contour_truncation=5.0,
    boundary_band_loss_weight=0.05, boundary_band_radius=3,
    gaussian_dense_weight=0.1, gaussian_node_weight=0.05,
    train_cfg=dict(), test_cfg=dict(mode='whole'))
