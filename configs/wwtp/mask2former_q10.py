_base_ = ['./mask2former.py']

# Every positive WWTP image holds exactly one target, so the COCO default of 100
# queries is oversized by an order of magnitude. The Hungarian matcher lights up
# a single query, and class_weight 0.1 leaves the other 99 unmatched queries
# carrying ~91% of the classification loss while their unsupervised masks still
# feed the inference einsum. At 10 queries the matched query carries 53% of that
# gradient instead of 9%.
model = dict(decode_head=dict(num_queries=10))

work_dir = 'work_dirs/wwtp/mask2former_q10'
