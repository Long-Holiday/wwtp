_base_ = [
    '../_base_/models/unet.py',
    '../_base_/datasets/wwtp_512x512.py',
    '../_base_/schedules/iter_40k.py',
    '../_base_/default_runtime.py',
]

train_dataloader = dict(batch_size=8)

