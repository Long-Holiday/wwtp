"""One model roster shared by training, evaluation, and smoke tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAINSTREAM_MODELS = (
    'hrnet', 'unetformer', 'segnext', 'segformer',
    'deeplabv3plus', 'mask2former', 'hdnet',
)
DATASETS = ('wwtp', 'uavid', 'loveda')
ALIASES = {'untformer': 'unetformer', 'deeplab': 'deeplabv3plus', 'hd_net': 'hdnet'}


def experiment_config(dataset, model):
    model = ALIASES.get(model, model)
    if dataset not in DATASETS or model not in MAINSTREAM_MODELS:
        raise ValueError(f'Unsupported experiment: {dataset}/{model}')
    directory = 'experiments' if dataset == 'wwtp' else f'remote_sensing/{dataset}'
    return ROOT / 'configs' / directory / f'{model}.py'
