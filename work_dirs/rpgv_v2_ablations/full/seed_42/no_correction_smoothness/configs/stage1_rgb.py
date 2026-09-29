auto_scale_lr = dict(base_batch_size=8, enable=False)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'
crop_size = (
    1024,
    1024,
)
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
data_root = 'wwtp_semantic_dataset'
dataset_type = 'WWTPDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=2000,
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
enable_early_stopping = False
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
metrics = [
    dict(
        iou_metrics=[
            'mIoU',
            'mDice',
            'mFscore',
        ],
        nan_to_num=0,
        type='IoUMetric'),
    dict(boundary_tolerance=3, pixel_size_m=0.5, type='BinaryBoundaryMetric'),
]
model = dict(
    coarse_loss_weight=0.3,
    component_cfg=dict(
        boundary_fusion=True,
        boundary_refinement=True,
        depth_rectification=True,
        detail_refinement=True,
        frequency_validation=True,
        learned_reliability=True,
        region_fusion=True,
        reliability_weighting=True),
    contour_truncation=10.0,
    correction_scale=0.1,
    correction_smoothness_weight=0.0,
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
    geometry_channels=[
        32,
        64,
        128,
        256,
    ],
    geometry_dropout_prob=0.15,
    global_thumbnail_size=512,
    loss_weights=dict(
        boundary=0.2,
        equivariance=0.05,
        final=1.0,
        geometry=0.2,
        preserve=0.05,
        reliability=0.05,
        rgb=0.3,
        sdf=0.1),
    max_correction_scale=0.25,
    max_logit_correction=2.0,
    max_offset=2.0,
    region_loss_weight=0.1,
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
    sdf_truncation=5.0,
    test_cfg=dict(crop_size=(
        1024,
        1024,
    ), mode='slide', stride=(
        768,
        768,
    )),
    train_cfg=dict(),
    training_stage='rgb',
    type='RPGVNetV2',
    use_contour=True,
    use_global_context=True,
    validation_channels=32)
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    accumulative_counts=8,
    clip_grad=dict(max_norm=10.0, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0006, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            norm=dict(decay_mult=0.0), rgb_encoder=dict(lr_mult=0.1)),
        norm_decay_mult=0.0),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=40000,
        eta_min=0.0,
        power=0.9,
        type='PolyLR'),
]
randomness = dict(deterministic=True, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations/test'),
        data_root='wwtp_semantic_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(size=(
                512,
                512,
            ), type='GenerateGlobalThumbnail'),
            dict(type='PackRPGVInputs'),
        ],
        type='WWTPDataset'),
    num_workers=2,
    persistent_workers=True,
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
    dict(boundary_tolerance=3, pixel_size_m=0.5, type='BinaryBoundaryMetric'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(size=(
        512,
        512,
    ), type='GenerateGlobalThumbnail'),
    dict(type='PackRPGVInputs'),
]
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=2000)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'),
        data_root='wwtp_semantic_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PhotoMetricDistortion'),
            dict(size=(
                512,
                512,
            ), type='GenerateGlobalThumbnail'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    1.5,
                ),
                scale=(
                    2048,
                    2048,
                ),
                type='RandomResize'),
            dict(
                degree=180,
                pad_val=0,
                prob=0.5,
                seg_pad_val=255,
                type='RandomRotate'),
            dict(
                crop_size=(
                    1024,
                    1024,
                ),
                edge_margin_ratio=0.15,
                foreground_label=1,
                foreground_prob=0.8,
                min_foreground_pixels=64,
                type='RandomForegroundCrop'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='RandomFlip'),
            dict(type='PackRPGVInputs'),
        ],
        type='WWTPDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PhotoMetricDistortion'),
    dict(size=(
        512,
        512,
    ), type='GenerateGlobalThumbnail'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            1.5,
        ),
        scale=(
            2048,
            2048,
        ),
        type='RandomResize'),
    dict(
        degree=180, pad_val=0, prob=0.5, seg_pad_val=255, type='RandomRotate'),
    dict(
        crop_size=(
            1024,
            1024,
        ),
        edge_margin_ratio=0.15,
        foreground_label=1,
        foreground_prob=0.8,
        min_foreground_pixels=64,
        type='RandomForegroundCrop'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='RandomFlip'),
    dict(type='PackRPGVInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/val', seg_map_path='annotations/val'),
        data_root='wwtp_semantic_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(size=(
                512,
                512,
            ), type='GenerateGlobalThumbnail'),
            dict(type='PackRPGVInputs'),
        ],
        type='WWTPDataset'),
    num_workers=2,
    persistent_workers=True,
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
    dict(boundary_tolerance=3, pixel_size_m=0.5, type='BinaryBoundaryMetric'),
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
work_dir = '/workspace/work_dirs/rpgv_v2_ablations/full/seed_42/no_correction_smoothness/stage1_rgb'
