#!/usr/bin/env python3
"""Prepare the official Potsdam normalized DSM as RPGV geometry patches.

The source is 1_DSM_normalisation.zip inside the official Potsdam.zip. Its
``normalized_lastools.jpg`` files are 6000x6000, single-channel, 8-bit DSM
rasters for all 38 tiles. They are dataset-supplied, but JPEG encoded; this
script does not claim to preserve the precision of the raw DSM TIFFs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / 'data/remote_sensing/prepared/potsdam'
SOURCE = REPO_ROOT / 'data/remote_sensing/raw/potsdam/Potsdam.zip'
OUTPUT = DATA_ROOT / 'dsm_geometry'
INNER_NAME = '1_DSM_normalisation.zip'
DSM_PATTERN = re.compile(
    r'^dsm_potsdam_(\d{2})_(\d{2})_normalized_lastools\.jpg$', re.I)
PATCH_PATTERN = re.compile(r'^(\d+_\d+)_x(\d+)_y(\d+)\.png$')
SPLITS = ('train', 'val', 'test')
TILE_SIZE = 6000


def patch_inventory(data_root: Path, patch_size: int, manifest: dict) -> dict:
    inventory = {}
    per_axis = math.ceil(TILE_SIZE / patch_size)
    expected_coords = {(x, y) for y in range(0, TILE_SIZE, patch_size)
                       for x in range(0, TILE_SIZE, patch_size)}
    split_tiles = {
        'train': manifest['train_tiles'],
        'val': manifest['internal_val_tiles'],
        'test': manifest['official_test_tiles'],
    }
    for split in SPLITS:
        by_tile = {tile: {} for tile in split_tiles[split]}
        image_dir = data_root / 'img_dir' / split
        for path in image_dir.glob('*.png'):
            match = PATCH_PATTERN.fullmatch(path.name)
            if not match or match.group(1) not in by_tile:
                raise ValueError(f'Unexpected Potsdam patch: {path}')
            tile, x, y = match.group(1), int(match.group(2)), int(match.group(3))
            if (x, y) in by_tile[tile]:
                raise ValueError(f'Duplicate patch coordinates: {path}')
            by_tile[tile][(x, y)] = path
        for tile, patches in by_tile.items():
            if set(patches) != expected_coords:
                raise ValueError(
                    f'{split}/{tile}: expected {per_axis * per_axis} aligned '
                    f'patches, found {len(patches)}')
            inventory[tile] = (split, patches)
    if len(inventory) != 38:
        raise ValueError(f'Expected 38 unique Potsdam tiles, found {len(inventory)}')
    return inventory


def dsm_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members = {}
    for info in archive.infolist():
        match = DSM_PATTERN.fullmatch(Path(info.filename).name)
        if match:
            tile = f'{int(match.group(1))}_{int(match.group(2))}'
            if tile in members or info.file_size == 0:
                raise ValueError(f'Duplicate or empty normalized DSM: {info.filename}')
            members[tile] = info
    return members


def valid_existing(path: Path, size: int) -> bool:
    try:
        with np.load(path) as data:
            return (set(data.files) == {'depth', 'reliability'}
                    and all(data[key].shape == (size, size)
                            and data[key].dtype == np.uint8
                            for key in ('depth', 'reliability')))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def save_patch(path: Path, depth: np.ndarray, reliability: np.ndarray) -> None:
    """Publish one patch atomically without replacing an existing file."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f'.{path.stem}.', suffix='.npz', delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, depth=depth, reliability=reliability)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--resume', action='store_true',
                        help='Continue an incomplete DSM output; never overwrite patches')
    parser.add_argument('--dry-run', action='store_true',
                        help='Check source and patch inventory without creating output')
    args = parser.parse_args()

    if args.output_root.resolve().is_relative_to(
        (args.data_root / 'pseudo_geometry').resolve()
    ):
        parser.error('DSM output must differ from the Depth Anything directory')
    if args.output_root.exists() and not (args.resume or args.dry_run):
        parser.error(f'Output exists; use --resume to validate and continue: {args.output_root}')

    manifest = json.loads((args.data_root / 'manifest.json').read_text())
    patch_size = int(manifest['patch_size'])
    inventory = patch_inventory(args.data_root, patch_size, manifest)
    with zipfile.ZipFile(args.source) as outer:
        matches = [info for info in outer.infolist()
                   if Path(info.filename).name == INNER_NAME]
        if len(matches) != 1:
            raise ValueError(f'Expected one {INNER_NAME} inside {args.source}')
        # Copy the nested ZIP to a temporary file once. Random access through
        # ZipExtFile would repeatedly scan the nearly 1 GB outer member.
        with tempfile.TemporaryDirectory(prefix='potsdam-dsm-') as temp_dir:
            inner_path = Path(temp_dir) / INNER_NAME
            with outer.open(matches[0]) as source, inner_path.open('xb') as dest:
                shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
            with zipfile.ZipFile(inner_path) as inner:
                members = dsm_members(inner)
                if set(members) != set(inventory):
                    raise ValueError(
                        f'DSM/RGB tile mismatch: missing={sorted(set(inventory)-set(members))}, '
                        f'extra={sorted(set(members)-set(inventory))}')
                counts = {split: 0 for split in SPLITS}
                for split, _ in inventory.values():
                    counts[split] += math.ceil(TILE_SIZE / patch_size) ** 2
                print(f'Official normalized DSM inventory: {len(members)} tiles; '
                      f'patches {counts}')
                if args.dry_run:
                    return

                metadata = {
                    'source': 'Potsdam.zip/1_DSM_normalisation.zip',
                    'variant': 'normalized_lastools.jpg',
                    'encoding': '8-bit grayscale JPEG supplied with official dataset',
                    'reliability': '255 inside tile, 0 in padded area',
                    'patch_size': patch_size,
                    'counts': counts,
                }
                metadata_path = args.output_root / 'manifest.json'
                if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
                    raise ValueError(f'Existing DSM manifest differs: {metadata_path}')
                for split in SPLITS:
                    (args.output_root / split).mkdir(parents=True, exist_ok=True)

                written = 0
                for tile in sorted(inventory):
                    split, patches = inventory[tile]
                    pending = [(coords, source, args.output_root / split / f'{source.stem}.npz')
                               for coords, source in sorted(patches.items())
                               if not (args.output_root / split / f'{source.stem}.npz').exists()]
                    for source in patches.values():
                        target = args.output_root / split / f'{source.stem}.npz'
                        if not target.exists():
                            continue
                        if not valid_existing(target, patch_size):
                            raise ValueError(f'Existing DSM patch invalid; refusing to overwrite: {target}')
                    if not pending:
                        print(f'{split}/{tile}: complete')
                        continue
                    with inner.open(members[tile]) as stream, Image.open(stream) as image:
                        if image.size != (TILE_SIZE, TILE_SIZE) or image.mode != 'L':
                            raise ValueError(f'Unexpected DSM image format: {members[tile].filename} '
                                             f'{image.size} {image.mode}')
                        image.load()
                        for (x, y), _, target in pending:
                            width = min(patch_size, TILE_SIZE - x)
                            height = min(patch_size, TILE_SIZE - y)
                            depth = np.zeros((patch_size, patch_size), dtype=np.uint8)
                            depth[:height, :width] = np.asarray(
                                image.crop((x, y, x + width, y + height)), dtype=np.uint8)
                            reliability = np.zeros_like(depth)
                            reliability[:height, :width] = 255
                            save_patch(target, depth, reliability)
                            written += 1
                    print(f'{split}/{tile}: wrote {len(pending)} patches')

                if not metadata_path.exists():
                    with metadata_path.open('x', encoding='utf-8') as stream:
                        json.dump(metadata, stream, indent=2, ensure_ascii=False)
                        stream.write('\n')
                print(f'Finished: wrote {written} patches to {args.output_root}')


if __name__ == '__main__':
    main()
