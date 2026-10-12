_base_ = ['./rpgv_v8.py']
# Drop the differentiable soft-contact dimension from the pair descriptor only;
# the relation network and every other objective stay identical.
model = dict(graph_cfg=dict(use_contact=False))
work_dir = 'work_dirs/wwtp/rpgv_v8_no_contact'
