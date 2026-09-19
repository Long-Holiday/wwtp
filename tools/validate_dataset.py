#!/usr/bin/env python3
"""Validate image/mask pairing and mask labels in the WWTP dataset."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', nargs='?', default='wwtp_semantic_dataset')
    parser.add_argument(
        '--limit', type=int, default=0,
        help='check at most N pairs per split; zero checks every pair')
    args = parser.parse_args()
    root = Path(args.data_root)

    failed = []
    for split in ('train', 'val', 'test'):
        images = {path.name: path for path in (root / 'images' / split).glob('*.png')}
        masks = {
            path.name: path
            for path in (root / 'annotations' / split).glob('*.png')
        }
        if images.keys() != masks.keys():
            failed.append(
                f'{split}: unpaired images={sorted(images.keys() - masks.keys())[:5]}, '
                f'masks={sorted(masks.keys() - images.keys())[:5]}')
        names = sorted(images.keys() & masks.keys())
        if args.limit:
            names = names[:args.limit]
        for name in names:
            try:
                image = np.asarray(Image.open(images[name]).convert('RGB'))
                # Keep palette-mode PNGs as class indices (do not convert RGB).
                mask = np.asarray(Image.open(masks[name]))
            except (OSError, ValueError):
                failed.append(f'{split}/{name}: unreadable image or mask')
                continue
            if image.shape[:2] != mask.shape[:2]:
                failed.append(
                    f'{split}/{name}: shape mismatch {image.shape} vs {mask.shape}')
            labels = set(np.unique(mask).tolist())
            if not labels.issubset({0, 1, 255}):
                failed.append(f'{split}/{name}: unexpected labels {sorted(labels)}')
        print(f'[OK] {split}: {len(images)} paired files; checked {len(names)} masks')
    if failed:
        raise SystemExit('\n'.join(failed))


if __name__ == '__main__':
    main()
