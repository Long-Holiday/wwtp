#!/usr/bin/env python3
"""Prepare LoveDA and Potsdam in a new, isolated directory from official ZIPs."""

import argparse
import json
import math
import re
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

RAW = Path('data/remote_sensing/raw')
OUT = Path('data/remote_sensing/prepared')
POTSDAM_TRAIN = set('''2_10 2_11 2_12 3_10 3_11 3_12 4_10 4_11 4_12
5_10 5_11 5_12 6_10 6_11 6_12 6_7 6_8 6_9 7_10 7_11 7_12 7_7 7_8 7_9'''.split())
POTSDAM_TEST = set('''5_15 6_15 6_13 3_13 4_14 6_14 5_14 2_13
4_15 2_14 5_13 4_13 3_14 7_13'''.split())
POTSDAM_VAL = {'2_10', '4_12', '6_7', '7_9'}
# Source RGB colors; raw label IDs 1..6 match MMSeg's PotsdamDataset.
POTSDAM_COLORS = {
    (0, 0, 0): 0, (255, 255, 255): 1, (0, 0, 255): 2,
    (0, 255, 255): 3, (0, 255, 0): 4, (255, 255, 0): 5,
    (255, 0, 0): 6,
}


def new_stage(target, resume=False):
    if target.exists():
        raise FileExistsError(f'Refusing to replace existing prepared data: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(target.name + '.building')
    if stage.exists():
        if resume:
            return stage
        raise FileExistsError(f'Previous staging directory exists; inspect it first: {stage}')
    stage.mkdir()
    return stage


def copy_zip_member(archive, member, target):
    if target.exists():
        raise ValueError(f'Duplicate filename in archive: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)


def prepare_loveda(raw, target):
    stage = new_stage(target)
    counts = {}
    for split in ('Train', 'Val', 'Test'):
        archive_path = raw / f'{split}.zip'
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith('.png'):
                    continue
                parts = Path(member.filename).parts
                if len(parts) < 4 or parts[0].lower() != split.lower():
                    continue
                if parts[-2] == 'images_png':
                    kind = 'img_dir'
                elif parts[-2] == 'masks_png' and split != 'Test':
                    kind = 'ann_dir'
                else:
                    continue
                if parts[-3] not in ('Urban', 'Rural'):
                    continue
                copy_zip_member(archive, member, stage / kind / split.lower() / parts[-1])
                counts[f'{split.lower()}_{kind}'] = counts.get(f'{split.lower()}_{kind}', 0) + 1
    expected = {'train_img_dir': 2522, 'train_ann_dir': 2522,
                'val_img_dir': 1669, 'val_ann_dir': 1669, 'test_img_dir': 1796}
    if counts != expected:
        raise ValueError(f'LoveDA archive counts differ from official split: {counts} != {expected}')
    for split in ('train', 'val'):
        imgs = {p.name for p in (stage / 'img_dir' / split).glob('*.png')}
        masks = {p.name for p in (stage / 'ann_dir' / split).glob('*.png')}
        if imgs != masks:
            raise ValueError(f'LoveDA {split} image/mask filenames differ')
    for split in ('train', 'val'):
        mask = next((stage / 'ann_dir' / split).glob('*.png'))
        with Image.open(mask) as image:
            sample = np.asarray(image)
        if sample.ndim != 2 or not set(np.unique(sample).tolist()).issubset(set(range(8))):
            raise ValueError(f'Unexpected LoveDA label encoding: {mask}')
    (stage / 'manifest.json').write_text(json.dumps({'source': 'Zenodo 5706578', 'counts': counts}, indent=2))
    stage.rename(target)
    print(f'Prepared LoveDA: {target} ({counts})')


@contextmanager
def nested_archive(outer, name, stage):
    matches = [m for m in outer.infolist() if Path(m.filename).name == name]
    if len(matches) != 1:
        raise ValueError(f'Expected one {name} in Potsdam.zip, found {len(matches)}')
    with tempfile.NamedTemporaryFile(dir=stage, suffix='.zip') as tmp:
        with outer.open(matches[0]) as src:
            shutil.copyfileobj(src, tmp, 8 * 1024 * 1024)
        tmp.flush()
        with zipfile.ZipFile(tmp.name) as archive:
            yield archive


def tile_members(archive):
    result = {}
    for member in archive.infolist():
        if member.is_dir() or not member.filename.lower().endswith(('.tif', '.tiff')):
            continue
        match = re.search(r'top_potsdam_(\d+_\d+)', Path(member.filename).name, re.I)
        if match:
            key = match.group(1)
            if key in result:
                raise ValueError(f'Duplicate Potsdam tile {key}')
            result[key] = member
    return result


def label_ids(image):
    rgb = np.asarray(image.convert('RGB'))
    packed = (rgb[:, :, 0].astype(np.uint32) << 16) | (rgb[:, :, 1].astype(np.uint32) << 8) | rgb[:, :, 2]
    values = np.unique(packed)
    mapping = {(r << 16) | (g << 8) | b: value for (r, g, b), value in POTSDAM_COLORS.items()}
    unknown = set(values.tolist()) - mapping.keys()
    if unknown:
        raise ValueError(f'Unexpected Potsdam label colors: {[hex(x) for x in sorted(unknown)[:10]]}')
    labels = np.zeros(packed.shape, dtype=np.uint8)
    for color, value in mapping.items():
        labels[packed == color] = value
    return Image.fromarray(labels, mode='L')


def save_patches(image, label, tile, split, stage, size):
    if image.size != label.size:
        raise ValueError(f'Potsdam image/mask shape mismatch: {tile}')
    width, height = image.size
    if (width, height) != (6000, 6000):
        raise ValueError(f'Unexpected Potsdam tile size {image.size}: {tile}')
    for y in range(0, height, size):
        for x in range(0, width, size):
            box = (x, y, min(x + size, width), min(y + size, height))
            name = f'{tile}_x{x:04d}_y{y:04d}.png'
            img_patch = Image.new('RGB', (size, size))
            img_patch.paste(image.crop(box))
            mask_patch = Image.new('L', (size, size), 0)  # 0 is ignore_index before reduce_zero_label
            mask_patch.paste(label.crop(box))
            img_patch.save(stage / 'img_dir' / split / name)
            mask_patch.save(stage / 'ann_dir' / split / name)


def save_eroded_test_masks(image, tile, stage, size):
    target = stage / 'ann_dir' / 'test_eroded'
    target.mkdir(parents=True, exist_ok=True)
    width, height = image.size
    if (width, height) != (6000, 6000):
        raise ValueError(f'Unexpected Potsdam noBoundary tile size {image.size}: {tile}')
    for y in range(0, height, size):
        for x in range(0, width, size):
            patch = Image.new('L', (size, size), 0)
            # Convert one patch at a time: a full-tile np.unique can exceed
            # the memory available alongside an active training process.
            patch.paste(label_ids(image.crop((x, y, min(x + size, width),
                                              min(y + size, height)))))
            patch.save(target / f'{tile}_x{x:04d}_y{y:04d}.png')


def prepare_potsdam(raw, target, size, resume=False):
    stage = new_stage(target, resume=resume)
    for split in ('train', 'val', 'test'):
        (stage / 'img_dir' / split).mkdir(parents=True, exist_ok=True)
        (stage / 'ann_dir' / split).mkdir(parents=True, exist_ok=True)
    official_eroded = False
    with zipfile.ZipFile(raw / 'Potsdam.zip') as outer:
        # The 4_12 image in Labels_all is a color overlay, not a class mask.
        # Historical participant labels are the clean 24 training references.
        with nested_archive(outer, '2_Ortho_RGB.zip', stage) as image_source, \
                nested_archive(outer, '5_Labels_for_participants.zip', stage) as train_labels, \
                nested_archive(outer, '5_Labels_all.zip', stage) as test_labels:
            images = tile_members(image_source)
            train_members = tile_members(train_labels)
            test_members = tile_members(test_labels)
            wanted = POTSDAM_TRAIN | POTSDAM_TEST
            if set(images) != wanted or set(train_members) != POTSDAM_TRAIN or not POTSDAM_TEST.issubset(test_members):
                raise ValueError('Potsdam image, participant-label, or complete-label tile inventory differs')
            per_tile = math.ceil(6000 / size) ** 2
            for i, tile in enumerate(sorted(wanted)):
                split = 'test' if tile in POTSDAM_TEST else 'val' if tile in POTSDAM_VAL else 'train'
                prior_images = list((stage / 'img_dir' / split).glob(f'{tile}_x*_y*.png'))
                prior_masks = list((stage / 'ann_dir' / split).glob(f'{tile}_x*_y*.png'))
                if len(prior_images) == len(prior_masks) == per_tile:
                    print(f'{i + 1}/38 {tile} -> {split} (already prepared)', flush=True)
                    continue
                if prior_images or prior_masks:
                    raise ValueError(f'Incomplete existing patches for {tile}; inspect staging directory')
                labels = test_labels if split == 'test' else train_labels
                member = test_members[tile] if split == 'test' else train_members[tile]
                with image_source.open(images[tile]) as src, labels.open(member) as lab:
                    image = Image.open(src).convert('RGB')
                    label = label_ids(Image.open(lab))
                    save_patches(image, label, tile, split, stage, size)
                print(f'{i + 1}/38 {tile} -> {split}', flush=True)
        eroded_archive_name = '5_Labels_all_noBoundary.zip'
        if any(Path(m.filename).name == eroded_archive_name for m in outer.infolist()):
            with nested_archive(outer, eroded_archive_name, stage) as eroded_source:
                eroded_labels = tile_members(eroded_source)
                if not POTSDAM_TEST.issubset(eroded_labels):
                    raise ValueError('Official noBoundary archive lacks Potsdam test labels')
                for tile in sorted(POTSDAM_TEST):
                    existing = list((stage / 'ann_dir' / 'test_eroded').glob(f'{tile}_x*_y*.png'))
                    if len(existing) == per_tile:
                        continue
                    if existing:
                        raise ValueError(f'Incomplete noBoundary patches for {tile}; inspect staging directory')
                    # TIFF decoders seek repeatedly. ZipExtFile emulates seeks
                    # by decompressing from the start, which is very slow here.
                    with tempfile.NamedTemporaryFile(dir=stage, suffix='.tif') as tmp:
                        with eroded_source.open(eroded_labels[tile]) as src:
                            shutil.copyfileobj(src, tmp, 8 * 1024 * 1024)
                        tmp.flush()
                        with Image.open(tmp.name) as image:
                            save_eroded_test_masks(image, tile, stage, size)
                official_eroded = True
    per_tile = math.ceil(6000 / size) ** 2
    for split, tiles in (('train', POTSDAM_TRAIN - POTSDAM_VAL),
                         ('val', POTSDAM_VAL), ('test', POTSDAM_TEST)):
        count = len(list((stage / 'img_dir' / split).glob('*.png')))
        mask_count = len(list((stage / 'ann_dir' / split).glob('*.png')))
        if count != len(tiles) * per_tile or mask_count != count:
            raise ValueError(f'Potsdam {split} patch count mismatch: {count} images, {mask_count} masks')
    if official_eroded and len(list((stage / 'ann_dir' / 'test_eroded').glob('*.png'))) != len(POTSDAM_TEST) * per_tile:
        raise ValueError('Potsdam official eroded reference count mismatch')
    manifest = {'source': 'ISPRS Potsdam official RGB, participant training labels, full test reference (2018)',
                'color_mode': 'RGB', 'patch_size': size,
                'train_tiles': sorted(POTSDAM_TRAIN - POTSDAM_VAL),
                'internal_val_tiles': sorted(POTSDAM_VAL),
                'official_test_tiles': sorted(POTSDAM_TEST),
                'official_eroded_reference': official_eroded}
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    stage.rename(target)
    print(f'Prepared Potsdam: {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['loveda', 'potsdam'])
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--patch-size', type=int, default=512)
    parser.add_argument('--resume', action='store_true', help='continue an existing .building directory')
    args = parser.parse_args()
    if args.patch_size <= 0 or args.patch_size > 6000:
        parser.error('--patch-size must be between 1 and 6000')
    if args.dataset == 'loveda':
        prepare_loveda(args.raw / 'loveda', args.output / 'loveda')
    else:
        prepare_potsdam(args.raw / 'potsdam', args.output / 'potsdam', args.patch_size, args.resume)


if __name__ == '__main__':
    main()
