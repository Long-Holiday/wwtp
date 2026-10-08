_base_ = ['../v5/rpgv_v5.py']
"""V6: joint frequency/region alignment, 1024 whole scenes, 5,000 updates."""

model = dict(
    type='RPGVNetV6',
    geometry_channels=[32, 64],
    rectifier_channels=16,
    max_offset=2.0,
    max_depth_correction=0.1,
    frequency_cutoff=0.15,
    frequency_padding=8,
    cluster_cfg=dict(
        grid_size=4, max_cluster_size=32, iterations=2,
        temperature=0.2, spatial_weight=2.0,
        min_reliability=0.35, min_region_mass=2.0, min_region_energy=1e-4),
    attention_cfg=dict(num_heads=4, query_size=32, kv_size=8),
    fusion_scale=0.1,
    contrastive_embed_dim=64,
    contrastive_temperature=0.07,
    contrastive_loss_weight=0.05,
    rectification_loss_weight=0.01,
    spatial_refinement=True,
    boundary_mode='band',
    boundary_band_loss_weight=0.05,
    boundary_band_radius=3,
)

optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={
        'rgb_encoder': dict(lr_mult=0.1),
        'logit_scale': dict(decay_mult=0.0),
    }))
load_from = None
resume = False
work_dir = 'work_dirs/rpgv_v6_1m'
