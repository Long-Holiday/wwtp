import os

_base_ = ['./loveda_segformer.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'

model = dict(
    _delete_=True,
    type='RPGVRemoteSensing',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512),
        test_cfg=dict(size_divisor=32)),
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
    num_classes=7,
    ignore_index=255,
    rgb_channels=(64, 128, 320, 512),
    decoder_channels=256,
    detail_channels=32,
    use_global_context=True,
    global_thumbnail_size=512,
    loss_weights=dict(final=1.0, coarse=0.4, aux=0.2, boundary=0.2),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)),
)
