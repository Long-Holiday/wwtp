# Historical diagnostics only; new training uses rpgv_v2_joint.py.
_base_ = ['./rpgv_stage1_rgb.py']

model = dict(
    type='RPGVNetV2', decoder_channels=64, detail_channels=24,
    contour_truncation=10.0, max_logit_correction=2.0,
    coarse_loss_weight=0.3, region_loss_weight=0.1,
    correction_smoothness_weight=0.02, use_contour=True)
# The training entry point otherwise injects min_delta=1.0, patience=3,
# which prematurely terminates comparisons after small but real improvements.
custom_hooks = [dict(type='EarlyStoppingHook', monitor='binary/Foreground_IoU',
                     rule='greater', min_delta=0.0, patience=20, strict=True)]
