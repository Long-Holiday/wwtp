#!/usr/bin/env python3
"""Build every experiment and run a tiny synthetic loss/metric pass."""

from __future__ import annotations

import argparse
import copy
import tempfile
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


def _add_pseudo_geometry(image: torch.Tensor) -> torch.Tensor:
    """Create deterministic, well-scaled geometry for RPGV smoke passes."""
    depth = image.mean(dim=0, keepdim=True)
    depth = 255.0 * (depth - depth.amin()) / (
        depth.amax() - depth.amin() + 1e-6)
    reliability = torch.full_like(depth, 192.0)
    return torch.cat([image, depth, reliability], dim=0)


def _check_rpgv_stage_freezing(model, stage: str) -> None:
    rgb_trainable = any(
        parameter.requires_grad for parameter in model.rgb_encoder.parameters())
    geometry_trainable = any(
        parameter.requires_grad for parameter in model.rectifier.parameters())
    decoder_trainable = any(
        parameter.requires_grad for parameter in model.decoder.parameters())
    expected = {
        'rgb': (True, False, True),
        'geometry': (False, True, False),
        'joint': (True, True, True),
    }[stage]
    actual = (rgb_trainable, geometry_trainable, decoder_trainable)
    if actual != expected:
        raise AssertionError(
            f'RPGV {stage} trainable groups are {actual}, expected {expected}')


def smoke_model(name: str, config_path: Path, size: int, backward: bool) -> None:
    cfg = Config.fromfile(config_path)
    import_modules_from_strings(**cfg.custom_imports)
    model_cfg = copy.deepcopy(cfg.model)
    _disable_pretraining(model_cfg)
    _small_rs_mamba(model_cfg)
    if model_cfg.get('type') == 'RPGVNet':
        model_cfg['global_thumbnail_size'] = size
    # A synthetic smoke pass should not pad a 64 px input back to 512 px.
    model_cfg['data_preprocessor']['size'] = (size, size)
    model = revert_sync_batchnorm(MODELS.build(model_cfg))
    model.train()
    if model_cfg.get('type') == 'RPGVNet':
        _check_rpgv_stage_freezing(
            model, model_cfg.get('training_stage', 'joint'))
    items = [_sample(size, index) for index in range(2)]
    if model_cfg.get('type') == 'RPGVNet':
        staged_items = []
        for image, sample in items:
            sample.global_img = PixelData(data=image.clone())
            if model_cfg.get('training_stage', 'joint') != 'rgb':
                image = _add_pseudo_geometry(image)
            staged_items.append((image, sample))
        items = staged_items
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
        if (
            model_cfg.get('type') == 'RPGVNet'
            and model_cfg.get('training_stage', 'joint') != 'rgb'
        ):
            reliability_grad = model.rectifier.learned_reliability[-1].weight.grad
            if reliability_grad is None or not reliability_grad.abs().sum():
                raise AssertionError(
                    f'{name}: reliability head received no gradient')
        if (
            model_cfg.get('type') == 'RPGVNet'
            and model_cfg.get('training_stage', 'joint') == 'joint'
        ):
            projection_grad = model.high_fusion.delta_projection[0].weight.grad
            if projection_grad is None or not projection_grad.abs().sum():
                raise AssertionError(
                    f'{name}: zero-init fusion projection received no gradient')
    if model_cfg.get('type') == 'RPGVNet':
        model.eval()
        with torch.no_grad():
            predictions = model.predict(**processed)
        prediction_shape = tuple(predictions[0].pred_sem_seg.data.shape)
        if prediction_shape != (1, size, size):
            raise AssertionError(
                f'RPGV prediction has shape {prediction_shape}, '
                f'expected {(1, size, size)}')
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad)
    print(
        f'[OK] {name:21s} params={parameters / 1e6:7.2f}M '
        f'trainable={trainable / 1e6:7.2f}M loss={total.item():.4f}')


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


def smoke_pseudo_geometry(size: int) -> None:
    """Check archive loading, five-channel rotation and Haar invertibility."""
    with tempfile.TemporaryDirectory() as temporary_dir:
        pseudo_root = Path(temporary_dir)
        split_dir = pseudo_root / 'train'
        split_dir.mkdir()
        depth = np.linspace(0.0, 1.0, size * size, dtype=np.float32).reshape(
            size, size)
        reliability = np.full((size, size), 0.75, dtype=np.float32)
        np.savez_compressed(
            split_dir / 'example.npz',
            depth=np.round(depth * 65535.0).astype(np.uint16),
            reliability=np.round(reliability * 255.0).astype(np.uint8))
        loader = TRANSFORMS.build(dict(
            type='LoadPseudoGeometry', pseudo_root=str(pseudo_root)))
        results = loader(dict(
            img=np.zeros((size, size, 3), dtype=np.uint8),
            img_path='/unused/train/example.png'))
        if results['img'].shape != (size, size, 5):
            raise AssertionError(
                'pseudo geometry was not appended as two channels')

        rotation = TRANSFORMS.build(dict(
            type='RandomRotate', prob=1.0, degree=10,
            pad_val=0, seg_pad_val=255))
        results.update(
            gt_seg_map=np.zeros((size, size), dtype=np.uint8),
            seg_fields=['gt_seg_map'])
        rotated = rotation(results)
        if rotated['img'].shape[-1] != 5:
            raise AssertionError('spatial augmentation dropped geometry channels')
        resize = TRANSFORMS.build(dict(
            type='RandomResize', scale=(2 * size, 2 * size),
            ratio_range=(1.0, 1.0), keep_ratio=True))
        crop = TRANSFORMS.build(dict(
            type='RandomForegroundCrop', crop_size=(size, size),
            foreground_prob=0.0))
        flip = TRANSFORMS.build(dict(
            type='RandomFlip', prob=1.0, direction='horizontal'))
        augmented = flip(crop(resize(rotated)))
        if augmented['img'].shape != (size, size, 5):
            raise AssertionError(
                'joint geometry augmentation produced '
                f'{augmented["img"].shape}')
        corruption = TRANSFORMS.build(dict(
            type='RandomPseudoGeometryCorruption', prob=1.0,
            modes=('zero',)))
        corrupted = corruption(augmented)
        if np.any(corrupted['img'][..., 3] != 0):
            raise AssertionError('forced zero-depth corruption did not run')
        if np.any(corrupted['pseudo_validity'] != 0):
            raise AssertionError(
                'zero-depth corruption did not create a zero validity target')

        thumbnail = TRANSFORMS.build(dict(
            type='GenerateGlobalThumbnail', size=(size // 2, size // 2)))
        packed_results = thumbnail(dict(
            img=np.zeros((size, size, 3), dtype=np.uint8),
            img_path='/unused/train/example.png',
            gt_seg_map=np.zeros((size, size), dtype=np.uint8),
            seg_fields=['gt_seg_map'],
            ori_shape=(size, size),
            img_shape=(size, size)))
        packer = TRANSFORMS.build(dict(type='PackRPGVInputs'))
        packed = packer(packed_results)
        global_shape = tuple(packed['data_samples'].global_img.data.shape)
        if global_shape != (3, size // 2, size // 2):
            raise AssertionError(f'global thumbnail has shape {global_shape}')
        validity_shape = tuple(
            packed['data_samples'].pseudo_validity.data.shape)
        if validity_shape != (1, size, size):
            raise AssertionError(
                f'pseudo validity target has shape {validity_shape}')

    from wwtpseg.models.utils import (
        BoundaryResidualRefiner,
        HaarWavelet2D,
        UncertaintyGatedResidualFusion,
    )
    wavelet = HaarWavelet2D()
    value = torch.randn(2, 3, size + 1, size - 1)
    low, high = wavelet(value)
    reconstructed = wavelet.inverse(low, high, value.shape[-2:])
    torch.testing.assert_close(reconstructed, value, rtol=1e-5, atol=1e-6)

    fusion = UncertaintyGatedResidualFusion(
        rgb_channels=8, geometry_channels=4, delta_channels=6,
        gate_channels=4)
    rgb = torch.randn(2, 8, size, size)
    fused, _ = fusion(
        rgb,
        torch.randn(2, 4, size // 2, size // 2),
        torch.randn(2, 6, size // 2, size // 2),
        torch.rand(2, 1, size, size),
        torch.rand(2, 1, size, size),
        torch.rand(2, 1, size, size))
    torch.testing.assert_close(fused, rgb, rtol=0.0, atol=0.0)

    refiner = BoundaryResidualRefiner(channels=8)
    torch.nn.init.normal_(refiner.refinement[-1].weight, std=0.02)
    decoder_feature = torch.randn(2, 8, size, size)
    without_geometry = refiner(
        decoder_feature, torch.zeros(2, 1, size, size))['final']
    with_geometry = refiner(
        decoder_feature, torch.rand(2, 1, size, size))['final']
    torch.testing.assert_close(
        with_geometry, without_geometry, rtol=0.0, atol=0.0)
    print(
        '[OK] pseudo-geometry augmentation, validity target, exact RGB '
        'initialization, global thumbnail and Haar DWT invariant')


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
    if 'rpgv_stage1_rgb' in args.models:
        smoke_dataset(EXPERIMENTS['rpgv_stage1_rgb'], args.data_root)
    smoke_location_augmentation()
    smoke_pseudo_geometry(args.input_size)
    smoke_metric(args.input_size)
    for name in args.models:
        smoke_model(name, EXPERIMENTS[name], args.input_size, args.backward)


if __name__ == '__main__':
    main()
