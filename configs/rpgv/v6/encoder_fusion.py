_base_ = ['rpgv_v6.py']

# Fusion-location control, not a reproduction of the historical v6 architecture.
model = dict(fusion_location='encoder')
work_dir = 'work_dirs/rpgv_v6_optimized_encoder_fusion_1m'
