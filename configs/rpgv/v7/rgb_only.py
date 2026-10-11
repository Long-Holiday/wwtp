_base_ = ['rpgv_v7.py']
model = dict(use_geometry=False)
train_pipeline = [step for step in _base_.train_pipeline if step['type'] != 'LoadPseudoGeometry']
test_pipeline = [step for step in _base_.test_pipeline if step['type'] != 'LoadPseudoGeometry']
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
work_dir = 'work_dirs/rpgv_v7_rgb_only_1m'
