import os

_base_ = ['./potsdam_segformer.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=1500, end=40000),
]
pseudo_root = os.getenv('POTSDAM_PSEUDO_ROOT', os.path.join(_base_.data_root, 'pseudo_geometry'))

custom_imports = dict(
    imports=['wwtpseg', 'wwtpseg.remote_sensing.model_full'],
    allow_failed_imports=False,
)

model = dict(
    _delete_=True,
    type='RPGVRemoteSensingFull',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        bgr_to_rgb=False,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512),
        test_cfg=dict(size_divisor=32),
    ),
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
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    ),
    num_classes=6,
    ignore_index=255,
    rgb_channels=(64, 128, 320, 512),
    geometry_channels=(32, 64, 128, 256),
    validation_channels=32,
    decoder_channels=256,
    detail_channels=32,
    correction_scale=0.1,
    max_correction_scale=0.25,
    max_offset=2.0,
    geometry_dropout_prob=0.15,
    use_global_context=True,
    global_thumbnail_size=512,
    loss_weights=dict(
        final=1.0,
        coarse=0.4,
        aux=0.2,
        geometry=0.2,
        boundary=0.2,
        preserve=0.05,
        equivariance=0.05,
        reliability=0.05,
    ),
    test_cfg=dict(mode='whole'),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateGlobalThumbnail', size=(512, 512)),
    dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
    dict(type='RandomRotate', prob=0.5, degree=180, pad_val=0, seg_pad_val=255),
    dict(
        type='RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal'],
    ),
    dict(type='RandomPseudoGeometryCorruption', prob=0.3),
    dict(type='PackRPGVInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='GenerateGlobalThumbnail', size=(512, 512)),
    dict(type='LoadPseudoGeometry', pseudo_root=pseudo_root, required=True),
    dict(type='PackRPGVInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type='PotsdamDataset',
                data_root=_base_.data_root,
                data_prefix=dict(
                    img_path='img_dir/train', seg_map_path='ann_dir/train'
                ),
                pipeline=train_pipeline,
            ),
            dict(
                type='PotsdamDataset',
                data_root=_base_.data_root,
                data_prefix=dict(
                    img_path='img_dir/val', seg_map_path='ann_dir/val'
                ),
                pipeline=train_pipeline,
            ),
        ],
    ),
)
val_dataloader = None
val_evaluator = None
val_cfg = None

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PotsdamDataset',
        data_root=_base_.data_root,
        data_prefix=dict(
            img_path='img_dir/test', seg_map_path='ann_dir/test'
        ),
        pipeline=test_pipeline,
    ),
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(
            rgb_encoder=dict(lr_mult=0.1),
            norm=dict(decay_mult=0.0),
        ),
        norm_decay_mult=0.0,
    ),
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000)
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        max_keep_ckpts=3,
    )
)
