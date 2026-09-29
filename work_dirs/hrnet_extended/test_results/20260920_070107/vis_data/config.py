auto_scale_lr = dict(base_batch_size=16, enable=False)
crop_size = (
    512,
    512,
)
custom_hooks = [
    dict(
        min_delta=1.0,
        monitor='binary/Foreground_IoU',
        patience=3,
        rule='greater',
        strict=False,
        type='EarlyStoppingHook'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'wwtpseg',
        'mmdet.models',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
data_root = '/workspace/wwtp_semantic_dataset'
dataset_type = 'WWTPDataset'
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
load_from = 'work_dirs/hrnet_extended/best_binary_Foreground_IoU_iter_7000.pth'
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
    backbone=dict(
        extra=dict(
            stage1=dict(
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_branches=1,
                num_channels=(64, ),
                num_modules=1),
            stage2=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                ),
                num_branches=2,
                num_channels=(
                    18,
                    36,
                ),
                num_modules=1),
            stage3=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                ),
                num_branches=3,
                num_channels=(
                    18,
                    36,
                    72,
                ),
                num_modules=4),
            stage4=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                    4,
                ),
                num_branches=4,
                num_channels=(
                    18,
                    36,
                    72,
                    144,
                ),
                num_modules=3)),
        init_cfg=dict(
            checkpoint='open-mmlab://msra/hrnetv2_w18', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        type='HRNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=270,
        concat_input=False,
        dropout_ratio=-1,
        in_channels=[
            18,
            36,
            72,
            144,
        ],
        in_index=(
            0,
            1,
            2,
            3,
        ),
        input_transform='resize_concat',
        kernel_size=1,
        loss_decode=[
            dict(
                avg_non_ignore=True,
                loss_name='loss_ce',
                loss_weight=1.0,
                type='CrossEntropyLoss'),
            dict(
                loss_name='loss_dice',
                loss_weight=1.0,
                type='DiceLoss',
                use_sigmoid=False),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=1,
        type='FCNHead'),
    test_cfg=dict(crop_size=(
        512,
        512,
    ), mode='slide', stride=(
        384,
        384,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0006, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(norm_decay_mult=0.0),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=3000,
        by_epoch=False,
        end=3500,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=3500,
        by_epoch=False,
        end=20000,
        eta_min=0.0,
        power=0.9,
        type='PolyLR'),
]
randomness = dict(deterministic=False, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations/test'),
        data_root='/workspace/wwtp_semantic_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
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
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=20000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'),
        data_root='/workspace/wwtp_semantic_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
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
                    512,
                    512,
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
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='WWTPDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
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
            512,
            512,
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
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/val', seg_map_path='annotations/val'),
        data_root='/workspace/wwtp_semantic_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
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
work_dir = 'work_dirs/hrnet_extended/test_results'
