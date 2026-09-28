_base_ = ['./rpgv_v4.py']
model = dict(use_global_context=True)
train_pipeline = [dict(step) for step in _base_.train_pipeline[:-1]]
train_pipeline.insert(3, dict(type='GenerateGlobalThumbnail', size=(512, 512)))
train_pipeline.append(dict(type='PackRPGVInputs'))
test_pipeline = [dict(step) for step in _base_.test_pipeline[:-1]]
test_pipeline.insert(2, dict(type='GenerateGlobalThumbnail', size=(512, 512)))
test_pipeline.append(dict(type='PackRPGVInputs'))
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
work_dir = 'work_dirs/rpgv_v4_global'
