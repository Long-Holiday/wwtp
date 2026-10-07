"""RPGV RGB stages share the v5 whole-scene protocol and retain global context."""
_base_ = ['./wwtp_1024x1024.py']

train_pipeline = [step for step in _base_.train_pipeline
                  if step['type'] not in ('RandomFlip', 'PackSegInputs')] + [
    dict(type='GenerateGlobalThumbnail', size=(512, 512)),
    dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackRPGVInputs')]
test_pipeline = [step for step in _base_.test_pipeline
                 if step['type'] != 'PackSegInputs'] + [
    dict(type='GenerateGlobalThumbnail', size=(512, 512)), dict(type='PackRPGVInputs')]
train_dataloader = dict(batch_size=1, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
