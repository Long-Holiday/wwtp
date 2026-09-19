norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512),
    test_cfg=dict(size_divisor=32))

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='RSMambaBackbone',
        in_channels=3,
        dims=96,
        depths=(2, 2, 9, 2),
        state_size=16,
        ssm_ratio=2.0,
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        use_checkpoint=True),
    decode_head=dict(
        type='FCNHead',
        in_channels=96,
        in_index=0,
        channels=96,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', loss_name='loss_ce',
                loss_weight=1.0, avg_non_ignore=True),
            dict(
                type='DiceLoss', use_sigmoid=False,
                loss_name='loss_dice', loss_weight=1.0),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))
