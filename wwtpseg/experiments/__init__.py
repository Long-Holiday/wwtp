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
OPTIONAL_MODELS = ('rpgv_v6',) + V6_ABLATIONS
EXPERIMENT_SUITES = {
    'baselines': MAINSTREAM_MODELS,
    'v6': ('rpgv_v6',),
    'v6-ablations': V6_ABLATIONS,
    'all': MAINSTREAM_MODELS + OPTIONAL_MODELS,
}
ALIASES = {'untformer': 'unetformer', 'deeplab': 'deeplabv3plus', 'hd_net': 'hdnet'}


def experiment_config(dataset, model):
    model = ALIASES.get(model, model)
    if dataset not in DATASETS:
        raise ValueError(f'Unsupported experiment: {dataset}/{model}')
    if model in V6_ABLATIONS:
        if dataset != 'wwtp':
            raise ValueError(f'V6 ablations are only available for WWTP: {dataset}/{model}')
        return ROOT / 'configs/rpgv/v6' / f'{model.removeprefix("rpgv_v6_")}.py'
    if model == 'rpgv_v6':
        if dataset == 'wwtp':
            return ROOT / 'configs/rpgv/v6/rpgv_v6.py'
    elif model not in MAINSTREAM_MODELS:
        raise ValueError(f'Unsupported experiment: {dataset}/{model}')
    return ROOT / 'configs' / dataset / f'{model}.py'
