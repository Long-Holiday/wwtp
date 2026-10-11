"""SegNeXt with FP32 LightHamHead under the shared v5 data/update budget."""
_base_ = [
    '../_base_/models/segnext_mscan_s.py',
    '../_base_/datasets/wwtp_1024x1024.py',
    '../_base_/schedules/baseline_20k.py',
    '../_base_/default_runtime.py',
]

work_dir = 'work_dirs/wwtp/segnext'
