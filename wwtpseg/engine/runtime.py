"""Runtime preparation shared by training, testing, and prediction."""
import torch
from pathlib import Path


def disable_pretraining(value):
    if isinstance(value, dict):
        if 'init_cfg' in value:
            value['init_cfg'] = None
        if 'pretrained' in value:
            value['pretrained'] = False
        for child in value.values():
            disable_pretraining(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            disable_pretraining(child)


def prepare_runtime(cfg, inference=False, no_pretrained=False):
    if cfg.get('benchmark_dataset'):
        from wwtpseg.datasets.public_protocol import validate_prepared
        validate_prepared(cfg.benchmark_dataset, cfg.data_root)
        for split in ('train', 'val', 'test'):
            dataset = cfg[f'{split}_dataloader'].dataset
            if Path(dataset.data_root) != Path(cfg.data_root):
                raise ValueError(f'{split} data_root differs from the validated official dataset')
        if cfg.get('predict_dataset') and Path(cfg.predict_dataset.data_root) != Path(cfg.data_root):
            raise ValueError('Prediction root differs from the validated official dataset')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if inference or no_pretrained or cfg.get('resume') or cfg.get('load_from'):
        disable_pretraining(cfg.model)
    if inference:
        cfg.resume = False
    # Preserve MMEngine's native resume and random-state behavior.
    return cfg
