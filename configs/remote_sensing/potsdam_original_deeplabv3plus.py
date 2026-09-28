_base_ = ['./potsdam_original_common.py', '../_base_/models/deeplabv3plus_r50.py']

model = dict(
    decode_head=dict(num_classes=6, loss_decode=dict(
        _delete_=True, type='CrossEntropyLoss', loss_weight=1.0, avg_non_ignore=True)),
    auxiliary_head=dict(num_classes=6),
    test_cfg=dict(mode='whole'),
)
