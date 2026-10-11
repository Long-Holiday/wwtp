_base_ = ['rpgv_v6.py']

model = dict(cluster_cfg=dict(max_cluster_size=32))
work_dir = 'work_dirs/rpgv_v6_siglip_cluster32_1m'
