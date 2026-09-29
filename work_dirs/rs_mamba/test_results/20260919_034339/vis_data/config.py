auto_scale_lr = dict(base_batch_size=16, enable=False)
crop_size = (
    512,
    512,
)
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
        interval=500,
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
load_from = 'work_dirs/rs_mamba/best_binary_Foreground_IoU_iter_3000.pth'
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
        depths=(
            2,
            2,
            9,
            2,
        ),
        dims=96,
        drop_path_rate=0.2,
        in_channels=3,
        mlp_ratio=4.0,
        ssm_ratio=2.0,
        state_size=16,
        type='RSMambaBackbone',
        use_checkpoint=True),
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
        channels=96,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=96,
        in_index=0,
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
    accumulative_counts=4,
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0002, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(a_logs=dict(decay_mult=0.0), ds=dict(decay_mult=0.0)),
        norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=300, start_factor=1e-06, type='LinearLR'),
    dict(
        begin=300,
        by_epoch=False,
        end=3000,
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
train_cfg = dict(max_iters=3000, type='IterBasedTrainLoop', val_interval=500)
train_dataloader = dict(
    batch_size=1,
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
work_dir = 'work_dirs/rs_mamba/test_results'
