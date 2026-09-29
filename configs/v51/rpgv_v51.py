"""V5.1: six-thousand-microstep refinement from the validation-selected v5."""
import os

_base_ = ['../v5/rpgv_v5.py']
model = dict(
    type='RPGVNetV51', spatial_refinement=True,
    boundary_mode='band', boundary_band_loss_weight=0.05,
    boundary_band_radius=3,
    rgb_encoder=dict(init_cfg=None))

# All inherited weights must be supplied by the strict migration utility.
load_from = os.getenv('RPGV_V51_INIT_CHECKPOINT')
required_previous_stage = 'Validated v5-to-v5.1 initialization'
resume = False
optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    paramwise_cfg=dict(custom_keys={
        'rgb_encoder': dict(lr_mult=0.1),
        'decoder.spatial8': dict(lr_mult=5.0),
        'decoder.spatial4': dict(lr_mult=5.0),
    }))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=400),
    dict(type='PolyLR', eta_min=0.0, power=0.9, by_epoch=False, begin=400, end=6000),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=6000, val_interval=500)
default_hooks = dict(checkpoint=dict(interval=500))
work_dir = 'work_dirs/rpgv_v51_1m'
