#!/usr/bin/env python3
"""Create a 1 m/px copy of the 0.5 m/px WWTP segmentation dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SOURCE_SIZE = (2048, 2048)
TARGET_SIZE = (1024, 1024)
SPLITS = ('train', 'val', 'test')


def resample_world_file(source: Path, target: Path) -> None:
    """Double pixel spacing while keeping the outer image bounds fixed."""
    values = [float(line) for line in source.read_text().splitlines()]
    if len(values) != 6:
        raise ValueError(f'expected six world-file values: {source}')
    a, d, b, e, x, y = values
    if abs(abs(a) - 0.5) > 1e-6 or abs(abs(e) - 0.5) > 1e-6:
        raise ValueError(f'expected 0.5 m source pixels: {source}')
    result = (2 * a, 2 * d, 2 * b, 2 * e,
              x + (a + b) / 2, y + (d + e) / 2)
    _atomic_write(target, ('\n'.join(f'{value:.15g}' for value in result) + '\n').encode())


def _atomic_write(target: Path, content: bytes) -> None:
    temporary = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
    try:
        temporary.write_bytes(content)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _save_png(image: Image.Image, target: Path) -> None:
    temporary = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
    try:
        image.save(temporary, format='PNG')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _convert_one(task: tuple[str, str, str, bool]) -> tuple[str, int, bool]:
    source_root, output_root, relative_name, with_geometry = task
    source, output = Path(source_root), Path(output_root)
    split, name = Path(relative_name).parts
    image_in = source / 'images' / split / name
    mask_in = source / 'annotations' / split / name
    image_out = output / 'images' / split / name
    mask_out = output / 'annotations' / split / name
    geometry_in = source / 'pseudo_geometry' / split / f'{Path(name).stem}.npz'
    geometry_out = output / 'pseudo_geometry' / split / geometry_in.name
    required = [image_out, mask_out, image_out.with_suffix('.pgw'),
                mask_out.with_suffix('.pgw')]
    if with_geometry:
        required.append(geometry_out)
    complete = all(path.is_file() for path in required)
    if not complete:
        with Image.open(image_in) as image:
            if image.size != SOURCE_SIZE or image.mode != 'RGB':
                raise ValueError(f'expected 2048x2048 RGB: {image_in}')
            resized_image = image.resize(TARGET_SIZE, Image.Resampling.BOX)
            _save_png(resized_image, image_out)
        with Image.open(mask_in) as mask:
            if mask.size != SOURCE_SIZE or mask.mode != 'P':
                raise ValueError(f'expected 2048x2048 palette mask: {mask_in}')
            resized_mask = mask.resize(TARGET_SIZE, Image.Resampling.NEAREST)
            resized_mask.putpalette(mask.getpalette())
            _save_png(resized_mask, mask_out)
        for origin, destination in ((image_in, image_out), (mask_in, mask_out)):
            resample_world_file(origin.with_suffix('.pgw'), destination.with_suffix('.pgw'))
        if with_geometry:
            with np.load(geometry_in) as archive:
                arrays = {}
                for key in ('depth', 'reliability'):
                    value = archive[key]
                    if value.shape != SOURCE_SIZE[::-1]:
                        raise ValueError(f'unexpected {key} shape: {geometry_in}')
                    arrays[key] = cv2.resize(
                        value, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            temporary = geometry_out.with_name(
                f'.{geometry_out.name}.{os.getpid()}.tmp')
            try:
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(geometry_out)
            finally:
                temporary.unlink(missing_ok=True)
    with Image.open(mask_out) as mask:
        labels = np.asarray(mask)
        if not np.isin(labels, (0, 1, 255)).all():
            raise ValueError(f'unexpected mask labels: {mask_out}')
        foreground = int(np.count_nonzero(labels == 1))
    return split, foreground, foreground > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, nargs='?',
                        default=Path('wwtp_semantic_dataset'))
    parser.add_argument('output', type=Path, nargs='?',
                        default=Path('wwtp_semantic_dataset_1m'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        raise ValueError('output must be separate from the source dataset')
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    stats = json.loads((source / 'dataset_stats.json').read_text())
    tasks = []
    geometry_counts = [len(list((source / 'pseudo_geometry' / split).glob('*.npz')))
                       for split in SPLITS]
    expected_counts = [len(list((source / 'images' / split).glob('*.png')))
                       for split in SPLITS]
    if any(geometry_counts) and geometry_counts != expected_counts:
        raise ValueError('pseudo geometry exists but is incomplete')
    with_geometry = any(geometry_counts)
    for split in SPLITS:
        names = sorted(path.name for path in (source / 'images' / split).glob('*.png'))
        if not names:
            raise ValueError(f'no images in {split}')
        masks = {path.name for path in (source / 'annotations' / split).glob('*.png')}
        if set(names) != masks:
            raise ValueError(f'image/mask mismatch in {split}')
        for kind in ('images', 'annotations'):
            (output / kind / split).mkdir(parents=True, exist_ok=True)
        if with_geometry:
            (output / 'pseudo_geometry' / split).mkdir(parents=True, exist_ok=True)
            geom = {path.stem for path in (source / 'pseudo_geometry' / split).glob('*.npz')}
            if geom != {Path(name).stem for name in names}:
                raise ValueError(f'image/geometry mismatch in {split}')
        tasks.extend((str(source), str(output), f'{split}/{name}', with_geometry)
                     for name in names)
    for name in ('classes.txt', 'palette.txt'):
        shutil.copy2(source / name, output / name)
    shutil.copytree(source / 'splits', output / 'splits', dirs_exist_ok=True)

    summary = {split: dict(images=0, foreground_pixels=0, positive_images=0)
               for split in SPLITS}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (split, foreground, positive) in enumerate(
                executor.map(_convert_one, tasks), start=1):
            summary[split]['images'] += 1
            summary[split]['foreground_pixels'] += foreground
            summary[split]['positive_images'] += int(positive)
            if index % 100 == 0 or index == len(tasks):
                print(f'converted {index}/{len(tasks)}', flush=True)

    stats['dataset_name'] = 'WWTP_Remote_Sensing_Semantic_Segmentation_1024_1m'
    stats['image_size'] = [1024, 1024]
    stats['resolution_m_per_px'] = 1.0
    stats['ground_coverage_m'] = 1024.0
    stats['source_dataset'] = str(source)
    stats['last_updated'] = datetime.now(timezone.utc).isoformat()
    stats['total_positive_images'] = 0
    stats['total_negative_images'] = 0
    for split, values in summary.items():
        count = values['images']
        positives = values['positive_images']
        stats['splits'][split].update(
            images=count,
            foreground_pixels=values['foreground_pixels'],
            foreground_coverage_pct=round(
                100 * values['foreground_pixels'] / (count * 1024 * 1024), 4),
            positive_images=positives,
            negative_images=count - positives,
        )
        stats['total_positive_images'] += positives
        stats['total_negative_images'] += count - positives
    stats['negative_sample_ratio'] = round(
        stats['total_negative_images'] / len(tasks), 4)
    _atomic_write(output / 'dataset_stats.json',
                  (json.dumps(stats, ensure_ascii=False, indent=2) + '\n').encode())
    readme = (
        '# WWTP 1 m/px 数据集\n\n'
        '由原始 2048×2048、0.5 m/px 数据集生成；影像和掩膜均为 '
        '1024×1024，单张地面覆盖仍为 1024 m × 1024 m。\n\n'
        'RGB 影像按 2×2 像素面积平均重采样；调色板掩膜按最近邻重采样，'
        '保留类别索引；伪几何 depth/reliability 按面积平均重采样。'
        '影像与掩膜的 `.pgw` 已同步更新至 1 m/px。'
        '目录结构、文件名和 train/val/test 划分与源数据集一致。\n\n'
        '训练时设置 `WWTP_DATA_ROOT=wwtp_semantic_dataset_1m` 和 '
        '`WWTP_IMAGE_SIZE=1024`。\n'
    )
    _atomic_write(output / 'README.md', readme.encode())
    print(f'completed: {output} ({len(tasks)} samples)', flush=True)


if __name__ == '__main__':
    main()
