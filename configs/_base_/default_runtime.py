default_scope = 'mmseg'
custom_imports = dict(
    imports=['wwtpseg', 'wwtpseg.models.segmentors.hdnet', 'mmdet.models'], allow_failed_imports=False)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=25, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=250,
        max_keep_ckpts=3,
        save_best='binary/Foreground_IoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
# LocalVisBackend keeps the text log; TensorboardVisBackend mirrors every
# LoggerHook scalar (train loss/lr/time, validation metrics) into the run's
# shared work_dir/<timestamp>/vis_data without touching the training protocol.
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)
