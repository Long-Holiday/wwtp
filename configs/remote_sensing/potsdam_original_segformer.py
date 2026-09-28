_base_ = ['./potsdam_original_common.py', '../_base_/models/segformer_mit_b2.py']

model = dict(
    decode_head=dict(num_classes=6, loss_decode=dict(
        _delete_=True, type='CrossEntropyLoss', loss_weight=1.0, avg_non_ignore=True)),
    test_cfg=dict(mode='whole'),
)
