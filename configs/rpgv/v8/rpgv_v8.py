_base_ = [
    '../../_base_/models/rpgv_net_v8.py',
    '../../_base_/datasets/wwtp_1024x1024.py',
    '../../_base_/schedules/baseline_20k.py',
    '../../_base_/default_runtime.py',
]

work_dir = 'work_dirs/wwtp/rpgv_v8'
