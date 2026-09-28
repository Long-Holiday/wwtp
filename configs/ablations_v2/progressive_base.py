_base_ = ['./full.py']
# RGB + raw geometry with direct, unweighted fusion. This is NOT RGB-only.
# Auxiliary coarse/SDF/RGB-boundary supervision remains common to all rows.
model = dict(
    component_cfg=dict(depth_rectification=False, learned_reliability=False,
                       frequency_validation=False, reliability_weighting=False),
    use_contour=False, region_loss_weight=0.0, final_boundary_loss_weight=0.0)
