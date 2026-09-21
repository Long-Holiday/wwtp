norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'

# RPGVNet performs RGB normalization internally because depth and reliability
# channels must remain in [0, 255] through the generic data preprocessor.
data_preprocessor = dict(
    type='SegDataPreProcessor',
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=(1024, 1024),
    test_cfg=dict(size_divisor=32))

model = dict(
    type='RPGVNet',
    data_preprocessor=data_preprocessor,
    rgb_encoder=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    rgb_channels=[64, 128, 320, 512],
    geometry_channels=[32, 64, 128, 256],
    validation_channels=32,
    decoder_channels=128,
    detail_channels=32,
    correction_scale=0.1,
    max_correction_scale=0.25,
    max_offset=2.0,
    # Joint training randomly disables geometry for whole samples so the
    # final decoder retains a supervised RGB-only fallback.
    geometry_dropout_prob=0.15,
    # The SDF head is stride 4: five feature pixels equal 20 input pixels.
    sdf_truncation=5.0,
    training_stage='joint',
    use_global_context=True,
    global_thumbnail_size=512,
    # Components remain instantiated when disabled so checkpoints are shared
    # across the full model and all controlled ablations.
    component_cfg=dict(
        depth_rectification=True,
        learned_reliability=True,
        frequency_validation=True,
        boundary_fusion=True,
        region_fusion=True,
        reliability_weighting=True,
        boundary_refinement=True,
        detail_refinement=True),
    loss_weights=dict(
        final=1.0,
        rgb=0.3,
        geometry=0.2,
        boundary=0.2,
        sdf=0.1,
        reliability=0.05,
        preserve=0.05,
        equivariance=0.05),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
