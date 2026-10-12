"""One model roster shared by training, evaluation, and smoke tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAINSTREAM_MODELS = (
    'hrnet', 'unetformer', 'segnext', 'segformer',
    'deeplabv3plus', 'mask2former', 'hdnet',
)
# Mask2Former's COCO query budget retuned for single-target WWTP imagery. Kept
# out of MAINSTREAM_MODELS: it exists for WWTP only, and the tests pin the
# baseline roster to seven names.
BASELINE_VARIANTS = ('mask2former_q10',)
DATASETS = ('wwtp', 'uavid', 'loveda', 'deepglobe')
EVALUATION_PROTOCOLS = {
    'wwtp': dict(metric='binary/Foreground_IoU', split='test', labeled_test=True),
    'uavid': dict(metric='mIoU', split='val', labeled_test=False),
    'loveda': dict(metric='mIoU', split='val', labeled_test=False),
    'deepglobe': dict(metric='roadIoU', split='test', labeled_test=True),
}
V6_ABLATIONS = (
    'rpgv_v6_rgb_only', 'rpgv_v6_no_contrastive', 'rpgv_v6_encoder_fusion',
    'rpgv_v6_legacy_contour', 'rpgv_v6_cluster32',
)
V7_ABLATIONS = (
    'rpgv_v7_rgb_only', 'rpgv_v7_fixed_grid', 'rpgv_v7_no_feedback',
    'rpgv_v7_no_relation_supervision', 'rpgv_v7_no_composition', 'rpgv_v7_no_structure_aux',
)
# More detailed objective/graph analyses are opt-in rather than default training jobs.
V7_ANALYSIS_ABLATIONS = (
    'rpgv_v7_no_hierarchical_supervision', 'rpgv_v7_no_node_supervision',
    'rpgv_v7_no_message_passing',
)
# Historical/specialized controls remain selectable with --models, outside default suites.
V7_LEGACY_ABLATIONS = (
    'rpgv_v7_no_contrastive', 'rpgv_v7_no_relational_supervision',
    'rpgv_v7_no_edges', 'rpgv_v7_no_perturbation', 'rpgv_v7_rgb_capacity',
)
V8_ABLATIONS = (
    'rpgv_v8_rgb_only', 'rpgv_v8_fixed_gaussians',
    'rpgv_v8_no_message_passing', 'rpgv_v8_no_gaussian_aux',
    'rpgv_v8_bounded_centers', 'rpgv_v8_no_contact', 'rpgv_v8_single_scale_dense',
)
# Model selection and offline geometry requirements are separate capabilities.
OFFLINE_GEOMETRY_MODELS = (
    ('rpgv_v6',) + V6_ABLATIONS + ('rpgv_v7',)
    + V7_ABLATIONS + V7_ANALYSIS_ABLATIONS + V7_LEGACY_ABLATIONS
)
OPTIONAL_MODELS = OFFLINE_GEOMETRY_MODELS + ('rpgv_v8',) + V8_ABLATIONS + BASELINE_VARIANTS
EXPERIMENT_SUITES = {
    'baselines': MAINSTREAM_MODELS,
    'v6': ('rpgv_v6',),
    'v6-ablations': V6_ABLATIONS,
    'all': MAINSTREAM_MODELS + ('rpgv_v6',) + V6_ABLATIONS,
    'v8': ('rpgv_v8',),
    'v8-ablations': V8_ABLATIONS,
    'v8-all': MAINSTREAM_MODELS + ('rpgv_v8',) + V8_ABLATIONS,
    'v7': ('rpgv_v7',),
    'v7-ablations': V7_ABLATIONS,
    'v7-all': MAINSTREAM_MODELS + ('rpgv_v6', 'rpgv_v7') + V7_ABLATIONS,
}
ALIASES = {'untformer': 'unetformer', 'deeplab': 'deeplabv3plus', 'hd_net': 'hdnet'}


def experiment_config(dataset, model):
    model = ALIASES.get(model, model)
    if dataset not in DATASETS:
        raise ValueError(f'Unsupported experiment: {dataset}/{model}')
    if model in V8_ABLATIONS:
        if dataset != 'wwtp':
            raise ValueError(f'V8 ablations are only available for WWTP: {dataset}/{model}')
        return ROOT / 'configs/rpgv/v8' / f'{model.removeprefix("rpgv_v8_")}.py'
    if (model in V6_ABLATIONS
            or model in V7_ABLATIONS + V7_ANALYSIS_ABLATIONS + V7_LEGACY_ABLATIONS):
        version = 'v6' if model in V6_ABLATIONS else 'v7'
        if dataset != 'wwtp':
            raise ValueError(f'{version.upper()} ablations are only available for WWTP: {dataset}/{model}')
        return ROOT / f'configs/rpgv/{version}' / f'{model.removeprefix(f"rpgv_{version}_")}.py'
    if model in BASELINE_VARIANTS:
        if dataset != 'wwtp':
            raise ValueError(f'Baseline variants are only available for WWTP: {dataset}/{model}')
        return ROOT / 'configs' / dataset / f'{model}.py'
    if model in ('rpgv_v6', 'rpgv_v7', 'rpgv_v8'):
        if dataset == 'wwtp':
            return ROOT / 'configs/rpgv' / model.removeprefix('rpgv_') / f'{model}.py'
    elif model not in MAINSTREAM_MODELS:
        raise ValueError(f'Unsupported experiment: {dataset}/{model}')
    return ROOT / 'configs' / dataset / f'{model}.py'
