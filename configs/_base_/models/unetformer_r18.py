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
        type='UNetFormerBackbone',
        backbone_name='resnet18.a1_in1k',
        pretrained=True,
        decode_channels=64,
        window_size=8,
        drop_path_rate=0.1),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=0,
        channels=64,
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
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=1,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', loss_weight=0.4,
            avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))
