#!/usr/bin/env python3
"""Build every experiment and run a tiny synthetic loss/metric pass."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmseg.registry import DATASETS, METRICS, MODELS, TRANSFORMS
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from mmseg.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    path.stem: path for path in sorted((ROOT / 'configs/experiments').glob('*.py'))
}


def _disable_pretraining(value) -> None:
    if isinstance(value, dict):
        if 'init_cfg' in value:
            value['init_cfg'] = None
        if 'pretrained' in value:
            value['pretrained'] = False
        for child in value.values():
            _disable_pretraining(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _disable_pretraining(child)


def _small_rs_mamba(model_cfg: dict) -> None:
    backbone = model_cfg.get('backbone', {})
    if backbone.get('type') != 'RSMambaBackbone':
        return
    backbone.update(
        dims=8,
        depths=(1, 1, 1, 1),
        state_size=4,
        ssm_ratio=1.0,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
    )
    model_cfg['decode_head'].update(in_channels=8, channels=8)


def _sample(size: int, offset: int = 0) -> tuple[torch.Tensor, SegDataSample]:
    generator = torch.Generator().manual_seed(42 + offset)
    image = torch.randint(
        0, 256, (3, size, size), generator=generator, dtype=torch.float32)
    mask = torch.zeros((1, size, size), dtype=torch.long)
    start, end = size // 4 + offset, 3 * size // 4
    mask[:, start:end, size // 4:3 * size // 4] = 1
    sample = SegDataSample()
    sample.set_metainfo(dict(
        img_shape=(size, size), ori_shape=(size, size),
        pad_shape=(size, size), scale_factor=(1.0, 1.0)))
    sample.gt_sem_seg = PixelData(data=mask)
    return image, sample


def smoke_model(name: str, config_path: Path, size: int, backward: bool) -> None:
    cfg = Config.fromfile(config_path)
    import_modules_from_strings(**cfg.custom_imports)
    model_cfg = copy.deepcopy(cfg.model)
    _disable_pretraining(model_cfg)
    _small_rs_mamba(model_cfg)
    # A synthetic smoke pass should not pad a 64 px input back to 512 px.
    model_cfg['data_preprocessor']['size'] = (size, size)
    model = revert_sync_batchnorm(MODELS.build(model_cfg))
    model.train()
    items = [_sample(size, index) for index in range(2)]
    batch = dict(
        inputs=[item[0] for item in items],
        data_samples=[item[1] for item in items])
    processed = model.data_preprocessor(batch, training=True)
    losses = model.loss(**processed)
    loss_terms = []
    for key, value in losses.items():
        if 'loss' not in key:
            continue
        loss_terms.extend(value if isinstance(value, (list, tuple)) else [value])
    total = sum(term.mean() for term in loss_terms)
    if not torch.isfinite(total):
        raise RuntimeError(f'{name}: non-finite loss {total.item()}')
    if backward:
        total.backward()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f'[OK] {name:14s} params={parameters / 1e6:7.2f}M loss={total.item():.4f}')


def smoke_metric(size: int) -> None:
    metric = METRICS.build(dict(
        type='BinaryBoundaryMetric', boundary_tolerance=1, pixel_size_m=0.5))
    _, sample = _sample(size)
    prediction = sample.gt_sem_seg.data.clone()
    sample.pred_sem_seg = PixelData(data=prediction)
    metric.process({}, [sample])
    values = metric.compute_metrics(metric.results)
    for key in ('Foreground_IoU', 'Dice', 'Boundary_F1'):
        if abs(values[key] - 100.0) > 1e-6:
            raise AssertionError(f'perfect prediction has {key}={values[key]}')
    if values['Hausdorff_px'] != 0.0:
        raise AssertionError('perfect prediction must have zero Hausdorff distance')
    print('[OK] binary/boundary metrics (perfect-mask invariant)')


def smoke_dataset(config_path: Path, data_root: Path) -> None:
    cfg = Config.fromfile(config_path)
    for split, dataloader_cfg in (
        ('train', cfg.train_dataloader), ('val', cfg.val_dataloader)
    ):
        dataset_cfg = copy.deepcopy(dataloader_cfg.dataset)
        dataset_cfg.data_root = str(data_root)
        dataset = DATASETS.build(dataset_cfg)
        if not dataset:
            raise AssertionError(f'{split} dataset is empty')
        item = dataset[0]
        image_shape = tuple(item['inputs'].shape)
        mask_shape = tuple(item['data_samples'].gt_sem_seg.data.shape)
        print(
            f'[OK] {split:5s} dataset len={len(dataset)} '
            f'image={image_shape} mask={mask_shape}')


def smoke_location_augmentation(crop_size: int = 512) -> None:
    """Verify that a source-centred target moves across the output crop."""
    transform = TRANSFORMS.build(dict(
        type='RandomForegroundCrop',
        crop_size=(crop_size, crop_size),
        foreground_prob=1.0,
        edge_margin_ratio=0.15,
        min_foreground_pixels=64))
    source_size = crop_size * 2
    image = np.zeros((source_size, source_size, 3), dtype=np.uint8)
    mask = np.zeros((source_size, source_size), dtype=np.uint8)
    half_object = max(crop_size // 16, 4)
    centre = source_size // 2
    mask[
        centre - half_object:centre + half_object,
        centre - half_object:centre + half_object,
    ] = 1

    np.random.seed(42)
    target_centres = []
    for _ in range(16):
        output = transform(dict(
            img=image.copy(),
            gt_seg_map=mask.copy(),
            seg_fields=['gt_seg_map']))
        coordinates = np.argwhere(output['gt_seg_map'] == 1)
        if not len(coordinates):
            raise AssertionError('foreground-aware crop discarded the target')
        target_centres.append(coordinates.mean(axis=0))

    target_centres = np.asarray(target_centres)
    spread_y, spread_x = np.ptp(target_centres, axis=0)
    if spread_y < crop_size * 0.3 or spread_x < crop_size * 0.3:
        raise AssertionError(
            'foreground target was not moved across enough crop positions: '
            f'y={spread_y:.1f}, x={spread_x:.1f}')
    print(
        '[OK] location augmentation '
        f'target-centre spread=(y={spread_y:.1f}, x={spread_x:.1f})')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--models', nargs='+', choices=sorted(EXPERIMENTS),
        default=sorted(EXPERIMENTS))
    parser.add_argument('--input-size', type=int, default=64)
    parser.add_argument('--backward', action='store_true')
    parser.add_argument(
        '--data-root', type=Path,
        default=ROOT / 'wwtp_semantic_dataset')
    args = parser.parse_args()
    if args.input_size < 32 or args.input_size % 32:
        raise ValueError('--input-size must be a multiple of 32 and at least 32')

    register_all_modules(init_default_scope=True)
    init_default_scope('mmseg')
    import wwtpseg  # noqa: F401

    smoke_dataset(next(iter(EXPERIMENTS.values())), args.data_root)
    smoke_location_augmentation()
    smoke_metric(args.input_size)
    for name in args.models:
        smoke_model(name, EXPERIMENTS[name], args.input_size, args.backward)


if __name__ == '__main__':
    main()
