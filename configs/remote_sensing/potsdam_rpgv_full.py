_base_ = ['./potsdam_rpgv.py']

# Final run: include all 24 historical training tiles. No checkpoint selection
# or model tuning on the 14 released benchmark test tiles.
train_dataloader = dict(dataset=dict(_delete_=True, type='ConcatDataset', datasets=[
    dict(type='PotsdamDataset', data_root=_base_.data_root,
         data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
         pipeline=_base_.train_pipeline),
    dict(type='PotsdamDataset', data_root=_base_.data_root,
         data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
         pipeline=_base_.train_pipeline),
]))
val_dataloader = None
val_evaluator = None
val_cfg = None
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000)
default_hooks = dict(checkpoint=dict(_delete_=True, type='CheckpointHook', by_epoch=False,
                                     interval=4000, max_keep_ckpts=3))
