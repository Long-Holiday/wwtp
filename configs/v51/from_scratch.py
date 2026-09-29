"""Independent ImageNet start for a later equal-budget architecture comparison."""
_base_ = ['./rpgv_v51.py']
model = dict(rgb_encoder=dict(init_cfg=dict(
    _delete_=True, type='Pretrained', checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth')))
load_from = None
required_previous_stage = None
optim_wrapper = dict(
    optimizer=dict(lr=6e-4),
    paramwise_cfg=dict(custom_keys=dict(
        _delete_=True, rgb_encoder=dict(lr_mult=0.1))))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type='PolyLR', eta_min=0.0, power=0.9, by_epoch=False, begin=1000, end=20000),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
default_hooks = dict(checkpoint=dict(interval=1000))
work_dir = 'work_dirs/rpgv_v51_from_scratch'
