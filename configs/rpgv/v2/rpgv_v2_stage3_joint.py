_base_ = ['../v1/rpgv_stage3_joint.py']
# Historical diagnostics only; new training uses rpgv_v2_joint.py.
import os


model = dict(
    type='RPGVNetV2', decoder_channels=64, detail_channels=24,
    contour_truncation=5.0, max_logit_correction=2.0,
    coarse_loss_weight=0.3, region_loss_weight=0.1,
    correction_smoothness_weight=0.02, use_contour=True)
load_from = os.getenv('RPGV_V2_STAGE2_CHECKPOINT') or None
required_previous_stage = 'v2 stage-2 geometry checkpoint (RPGV_V2_STAGE2_CHECKPOINT)'
custom_hooks = [dict(type='EarlyStoppingHook', monitor='binary/Foreground_IoU',
                     rule='greater', min_delta=0.0, patience=20, strict=True)]
