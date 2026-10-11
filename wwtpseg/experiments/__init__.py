"""One model roster shared by training, evaluation, and smoke tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAINSTREAM_MODELS = (
    'hrnet', 'unetformer', 'segnext', 'segformer',
    'deeplabv3plus', 'mask2former', 'hdnet',
)
DATASETS = ('wwtp', 'uavid', 'loveda')
V6_ABLATIONS = (
    'rpgv_v6_rgb_only', 'rpgv_v6_no_contrastive', 'rpgv_v6_encoder_fusion',
    'rpgv_v6_legacy_contour', 'rpgv_v6_cluster32',
)
V7_ABLATIONS = (
    'rpgv_v7_rgb_only', 'rpgv_v7_no_contrastive', 'rpgv_v7_no_relational_supervision',
    'rpgv_v7_no_edges', 'rpgv_v7_no_perturbation', 'rpgv_v7_no_feedback',
    'rpgv_v7_fixed_grid', 'rpgv_v7_rgb_capacity',
)
OPTIONAL_MODELS = ('rpgv_v6',) + V6_ABLATIONS + ('rpgv_v7',) + V7_ABLATIONS
EXPERIMENT_SUITES = {
    'baselines': MAINSTREAM_MODELS,
    'v6': ('rpgv_v6',),
    'v6-ablations': V6_ABLATIONS,
    'all': MAINSTREAM_MODELS + ('rpgv_v6',) + V6_ABLATIONS,
    'v7': ('rpgv_v7',),
    'v7-ablations': V7_ABLATIONS,
    'v7-all': MAINSTREAM_MODELS + ('rpgv_v6', 'rpgv_v7') + V7_ABLATIONS,
}
ALIASES = {'untformer': 'unetformer', 'deeplab': 'deeplabv3plus', 'hd_net': 'hdnet'}


def experiment_config(dataset, model):
    model = ALIASES.get(model, model)
    if dataset not in DATASETS:
        raise ValueError(f'Unsupported experiment: {dataset}/{model}')
    if model in V6_ABLATIONS or model in V7_ABLATIONS:
        version = 'v6' if model in V6_ABLATIONS else 'v7'
        if dataset != 'wwtp':
            raise ValueError(f'{version.upper()} ablations are only available for WWTP: {dataset}/{model}')
        return ROOT / f'configs/rpgv/{version}' / f'{model.removeprefix(f"rpgv_{version}_")}.py'
    if model in ('rpgv_v6', 'rpgv_v7'):
        if dataset == 'wwtp':
            return ROOT / 'configs/rpgv' / model.removeprefix('rpgv_') / f'{model}.py'
    elif model not in MAINSTREAM_MODELS:
        raise ValueError(f'Unsupported experiment: {dataset}/{model}')
    return ROOT / 'configs' / dataset / f'{model}.py'
