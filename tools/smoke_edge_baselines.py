#!/usr/bin/env python3
"""Check isolated edge baselines with synthetic masks; no dataset writes."""

from __future__ import annotations

import copy
import argparse
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.structures import PixelData
from mmengine.utils import import_modules_from_strings
from mmseg.registry import DATASETS, MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]


def sample(size: int, foreground: bool = True) -> SegDataSample:
    item = SegDataSample()
    item.set_metainfo(dict(img_shape=(size, size), ori_shape=(size, size),
                           pad_shape=(size, size),
                           padding_size=[0, 0, 0, 0]))
    mask = torch.zeros((1, size, size), dtype=torch.long)
    if foreground:
        mask[:, size // 4:3 * size // 4,
             size // 4:3 * size // 4] = 1
    mask[:, :3, :3] = 255
    item.gt_sem_seg = PixelData(data=mask)
    return item


def check(name: str) -> None:
    cfg = Config.fromfile(ROOT / 'configs' / 'edge_baselines' /
                          f'{name}.py')
    import_modules_from_strings(**cfg.custom_imports)
    model_cfg = copy.deepcopy(cfg.model)
    if name == 'cbr_net':
        model_cfg.pretrained = False
    model_cfg.data_preprocessor.size = (64, 64)
    model_cfg.test_cfg = dict(mode='slide', crop_size=(64, 64),
                              stride=(48, 48))
    model = revert_sync_batchnorm(MODELS.build(model_cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    batch = dict(inputs=[torch.randint(0, 256, (3, 64, 64)).float()
                         for _ in range(2)],
                 data_samples=[sample(64), sample(64, foreground=False)])
    processed = model.data_preprocessor(batch, training=True)
    losses = model.loss(**processed)
    total = sum(value for key, value in losses.items() if 'loss' in key)
    assert torch.isfinite(total), f'{name}: non-finite loss'
    total.backward()
    gradients = [p.grad for p in model.network.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in gradients), \
        f'{name}: no gradient'
    if name == 'cbr_net':
        assert model.network.direction_head[-1].weight.grad is not None
    else:
        assert model.network.stages[0].flow.weight.grad is not None
    model.eval()
    with torch.no_grad():
        images = torch.randint(0, 256, (1, 3, 96, 96)).float()
        test_batch = model.data_preprocessor(
            dict(inputs=[images[0]], data_samples=[sample(96)]),
            training=False)
        result = model.predict(**test_batch)
    assert tuple(result[0].pred_sem_seg.data.shape) == (1, 96, 96)
    print(f'[OK] {name}: loss={total.item():.4f}, backward, '
          '96x96 sliding prediction')


def check_dataset(name: str) -> None:
    cfg = Config.fromfile(ROOT / 'configs' / 'edge_baselines' /
                          f'{name}.py')
    import_modules_from_strings(**cfg.custom_imports)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    item = dataset[0]
    mask = item['data_samples'].gt_sem_seg.data
    labels = set(mask.unique().tolist())
    assert labels <= {0, 1, 255}, f'{name}: unexpected labels {labels}'
    assert item['inputs'].shape[0] == 3
    print(f'[OK] {name}: read-only training sample '
          f'image={tuple(item["inputs"].shape)}, labels={sorted(labels)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', action='store_true',
                        help='also read one item through each training pipeline')
    args = parser.parse_args()
    register_all_modules(init_default_scope=False)
    init_default_scope('mmseg')
    for experiment in ('cbr_net', 'hd_net'):
        check(experiment)
        if args.dataset:
            check_dataset(experiment)
