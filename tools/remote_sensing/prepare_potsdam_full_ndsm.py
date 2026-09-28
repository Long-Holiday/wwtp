#!/usr/bin/env python3
"""Align the official Potsdam normalized DSM (nDSM) to the experiment grid."""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

from wwtpseg.remote_sensing.potsdam_original import SPLIT_TILES


NDSM_PATTERN = re.compile(
    r'dsm_potsdam_(\d{2})_(\d{2})_normalized_lastools\.jpg$', re.I)


def source_inventory(source: Path) -> dict[str, Path]:
    inventory = {}
    for path in source.glob('*_normalized_lastools.jpg'):
        match = NDSM_PATTERN.fullmatch(path.name)
        if not match:
            continue
        tile = f'{int(match.group(1))}_{int(match.group(2))}'
        if tile in inventory:
            raise ValueError(f'Duplicate nDSM tile: {tile}')
        inventory[tile] = path
    if len(inventory) != 38:
        raise ValueError(f'Expected 38 official nDSM tiles, found {len(inventory)}')
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path('data/remote_sensing/prepared/potsdam'))
    parser.add_argument('--size', type=int, default=1536)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error('--size must be positive')

    source = (args.data_root / '1_DSM_normalisation'
              / '1_DSM_normalisation')
    inventory = source_inventory(source)
    output = args.data_root / f'geometry_official_ndsm_{args.size}'
    output.mkdir(parents=True, exist_ok=True)
    tiles = sorted(set().union(*SPLIT_TILES.values()))

    for index, tile in enumerate(tiles, 1):
        target = output / f'top_potsdam_{tile}_RGB.npz'
        if target.exists():
            with np.load(target) as existing:
                if (set(existing.files) != {'depth', 'reliability'}
                        or existing['depth'].shape != (args.size, args.size)
                        or existing['depth'].dtype != np.uint8
                        or existing['reliability'].shape != (args.size, args.size)
                        or existing['reliability'].dtype != np.uint8):
                    raise ValueError(f'Invalid existing nDSM geometry: {target}')
            print(f'[{index}/{len(tiles)}] {tile}: existing', flush=True)
            continue

        ndsm = cv2.imread(str(inventory[tile]), cv2.IMREAD_GRAYSCALE)
        if ndsm is None or ndsm.shape != (6000, 6000):
            raise ValueError(
                f'Unexpected nDSM tile: {inventory[tile]}, '
                f'{None if ndsm is None else ndsm.shape}')
        depth = cv2.resize(ndsm, (args.size, args.size),
                           interpolation=cv2.INTER_AREA)
        reliability = np.full(depth.shape, 255, dtype=np.uint8)
        temporary = output / f'.{target.name}.tmp.npz'
        try:
            np.savez_compressed(temporary, depth=depth,
                                reliability=reliability)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'[{index}/{len(tiles)}] {tile}', flush=True)

    metadata = dict(
        source='Potsdam/1_DSM_normalisation/normalized_lastools.jpg',
        variant='official normalized DSM (nDSM), 8-bit grayscale JPEG',
        input_shape=[6000, 6000],
        output_shape=[args.size, args.size],
        resampling='OpenCV INTER_AREA; source 0-255 values preserved',
        reliability='255 for every source pixel',
        tiles={split: sorted(values) for split, values in SPLIT_TILES.items()},
    )
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError(f'Existing manifest differs: {manifest}')
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Official nDSM geometry prepared: {output} ({len(tiles)} tiles)')


if __name__ == '__main__':
    main()
