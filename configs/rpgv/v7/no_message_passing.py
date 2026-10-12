_base_ = ['no_composition.py']
# Compare ONLY with no_composition: replace message passing by independent node MLPs.
# Unordered node pooling cannot discriminate composition permutations; keep relation supervision.
model = dict(graph_cfg=dict(use_edges=False))
work_dir = 'work_dirs/rpgv_v7_no_message_passing_1m'
