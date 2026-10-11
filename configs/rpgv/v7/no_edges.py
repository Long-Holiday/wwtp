_base_ = ['no_relational_supervision.py']
# Paired with no_relational_supervision: only replace graph message passing by node MLPs.
# Composition ranking cannot discriminate permutations of an unordered bag of nodes.
model = dict(graph_cfg=dict(use_edges=False))
work_dir = 'work_dirs/rpgv_v7_no_edges_1m'
