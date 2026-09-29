auto_scale_lr = dict(base_batch_size=8, enable=False)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'
custom_hooks = []
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'wwtpseg',
        'mmdet.models',
    ])
data_preprocessor = dict(
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=(
        1024,
        1024,
    ),
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
data_root = 'wwtp_semantic_dataset_1m'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=3,
        rule='greater',
        save_best='binary/Foreground_IoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    coarse_loss_weight=0.3,
    contour_truncation=5.0,
    data_preprocessor=dict(
        bgr_to_rgb=False,
        pad_val=0,
        seg_pad_val=255,
        size=(
            1024,
            1024,
        ),
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'),
    decoder_channels=64,
    detail_channels=24,
    final_boundary_loss_weight=0.1,
    geometry_channels=[
        16,
        32,
    ],
    geometry_dropout_prob=0.1,
    geometry_input='depth',
    max_logit_correction=2.0,
    region_loss_weight=0.05,
    rgb_channels=[
        64,
        128,
        320,
        512,
    ],
    rgb_encoder=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=64,
        in_channels=3,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            3,
            4,
            6,
            3,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
    sdf_loss_weight=0.1,
    test_cfg=dict(mode='whole'),
    type='RPGVNetV5',
    use_context=True,
    use_contour=True,
    use_geometry=True)
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    accumulative_counts=4,
    clip_grad=dict(max_norm=10.0, norm_type=2),
    loss_scale=dict(init_scale=128.0),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0006, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(rgb_encoder=dict(lr_mult=0.1)), norm_decay_mult=0.0),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.01, type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=20000,
        eta_min=0.0,
        power=0.9,
        type='PolyLR'),
]
pseudo_root = 'wwtp_semantic_dataset_1m/pseudo_geometry'
randomness = dict(deterministic=False, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations/test'),
        data_root='wwtp_semantic_dataset_1m',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                pseudo_root='wwtp_semantic_dataset_1m/pseudo_geometry',
                required=True,
                type='LoadPseudoGeometry'),
            dict(type='PackSegInputs'),
        ],
        type='WWTPDataset'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        iou_metrics=[
            'mIoU',
            'mDice',
            'mFscore',
        ],
        nan_to_num=0,
        type='IoUMetric'),
    dict(
        boundary_tolerance=1.5, pixel_size_m=1.0, type='BinaryBoundaryMetric'),
    dict(small_area=64, type='BinaryTopologyMetric'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        pseudo_root='wwtp_semantic_dataset_1m/pseudo_geometry',
        required=True,
        type='LoadPseudoGeometry'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=20000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'),
        data_root='wwtp_semantic_dataset_1m',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PhotoMetricDistortion'),
            dict(
                pseudo_root='wwtp_semantic_dataset_1m/pseudo_geometry',
                required=True,
                type='LoadPseudoGeometry'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='WWTPDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PhotoMetricDistortion'),
    dict(
        pseudo_root='wwtp_semantic_dataset_1m/pseudo_geometry',
        required=True,
        type='LoadPseudoGeometry'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='RandomFlip'),
    dict(type='PackSegInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/val', seg_map_path='annotations/val'),
        data_root='wwtp_semantic_dataset_1m',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                pseudo_root='wwtp_semantic_dataset_1m/pseudo_geometry',
                required=True,
                type='LoadPseudoGeometry'),
            dict(type='PackSegInputs'),
        ],
        type='WWTPDataset'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        iou_metrics=[
            'mIoU',
            'mDice',
            'mFscore',
        ],
        nan_to_num=0,
        type='IoUMetric'),
    dict(
        boundary_tolerance=1.5, pixel_size_m=1.0, type='BinaryBoundaryMetric'),
    dict(small_area=64, type='BinaryTopologyMetric'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/rpgv_v5_1m'
