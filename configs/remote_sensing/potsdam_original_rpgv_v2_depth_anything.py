"""RPGV v2 with offline Depth Anything V2 geometry on full Potsdam tiles."""

import os

_base_ = ['./potsdam_original_common.py']

geometry_root = os.getenv(
    'POTSDAM_PSEUDO_ROOT',
    os.path.join(_base_.data_root, 'geometry_depth_anything_1536'))

model = dict(
    type='RPGVRemoteSensingV2',
    data_preprocessor=dict(
        type='SegDataPreProcessor', bgr_to_rgb=False,
        pad_val=0, seg_pad_val=255, size=(1536, 1536),
        test_cfg=dict(size_divisor=32)),
    rgb_encoder=dict(
        type='MixVisionTransformer', in_channels=3, embed_dims=64,
        num_stages=4, num_layers=[3, 4, 6, 3], num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3], sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=(
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
            'segformer/mit_b2_20220624-66e8bf70.pth'))),
    num_classes=6, ignore_index=255,
    rgb_channels=(64, 128, 320, 512),
    geometry_channels=(32, 64, 128, 256),
    validation_channels=32, decoder_channels=64, detail_channels=24,
    correction_scale=0.1, max_correction_scale=0.25,
    max_offset=2.0, geometry_dropout_prob=0.15,
    use_global_context=True, global_thumbnail_size=256,
    max_logit_correction=2.0,
    loss_weights=dict(final=1.0, coarse=0.4, aux=0.2, geometry=0.2,
                      boundary=0.2, preserve=0.05, equivariance=0.05,
                      reliability=0.05),
    test_cfg=dict(mode='whole'),
)

train_pipeline = [
    dict(type='LoadPotsdamRGB', size=1536),
    dict(type='LoadPotsdamColorAnnotations'),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateGlobalThumbnail', size=(256, 256)),
    dict(type='LoadPseudoGeometry', pseudo_root=geometry_root, required=True),
    dict(type='RandomRotate', prob=0.5, degree=180, pad_val=0, seg_pad_val=255),
    dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomPseudoGeometryCorruption', prob=0.3),
    dict(type='PackRPGVInputs'),
]
test_pipeline = [
    dict(type='LoadPotsdamRGB', size=1536),
    dict(type='LoadPotsdamColorAnnotations'),
    dict(type='SetPotsdamEvaluationShape'),
    dict(type='GenerateGlobalThumbnail', size=(256, 256)),
    dict(type='LoadPseudoGeometry', pseudo_root=geometry_root, required=True),
    dict(type='PackRPGVInputs'),
]

train_dataloader = dict(dataset=dict(
    geometry_root=geometry_root, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(
    geometry_root=geometry_root, pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(
    geometry_root=geometry_root, pipeline=test_pipeline))
