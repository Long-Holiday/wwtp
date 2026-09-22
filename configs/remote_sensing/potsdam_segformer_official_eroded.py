_base_ = ['./potsdam_segformer_full.py']

# Evaluate the same fixed checkpoint against the official noBoundary labels.
test_dataloader = dict(dataset=dict(data_prefix=dict(
    img_path='img_dir/test', seg_map_path='ann_dir/test_eroded')))
test_evaluator = [dict(type='PotsdamFiveClassMetric', erode_radius=0,
                       prefix='official_eroded')]
