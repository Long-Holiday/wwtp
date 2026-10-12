_base_ = [
    '../_base_/datasets/deepglobe_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/public_runtime.py',
]

model = dict(
    type='HDNetBaseline', base_channel=48, num_classes=2,
    data_preprocessor=dict(
        type='SegDataPreProcessor', mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375], bgr_to_rgb=True,
        pad_val=0, seg_pad_val=255, size=(1024, 1024),
        test_cfg=dict(size_divisor=32)),
    test_cfg=dict(mode='whole'))
default_hooks = dict(checkpoint=dict(save_best='roadIoU', rule='greater'))
work_dir = 'work_dirs/deepglobe/hdnet'
