_base_ = ['../rpgv_v3_adapter.py']
# Same trainable capacity and RGB anchor; RGB intensity replaces depth and Q=1.
model = dict(use_geometry_evidence=False, geometry_dropout_prob=0.0)
work_dir = 'work_dirs/rpgv_v3_rgb_adapter_control'
