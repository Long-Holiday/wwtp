"""One model roster shared by training, evaluation, and smoke tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAINSTREAM_MODELS = (
    'hrnet', 'unetformer', 'segnext', 'segformer',
    'deeplabv3plus', 'mask2former', 'hdnet',
)
DATASETS = ('wwtp', 'uavid', 'loveda')
OPTIONAL_MODELS = ('rpgv_v6',)
ALIASES = {'untformer': 'unetformer', 'deeplab': 'deeplabv3plus', 'hd_net': 'hdnet'}


def experiment_config(dataset, model):
    model = ALIASES.get(model, model)
    if dataset not in DATASETS:
        raise ValueError(f'Unsupported experiment: {dataset}/{model}')
    if model == 'rpgv_v6':
        if dataset == 'wwtp':
            return ROOT / 'configs/rpgv/v6/rpgv_v6.py'
    elif model not in MAINSTREAM_MODELS:
        raise ValueError(f'Unsupported experiment: {dataset}/{model}')
    return ROOT / 'configs' / dataset / f'{model}.py'
