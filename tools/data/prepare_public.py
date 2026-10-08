#!/usr/bin/env python3
"""Prepare already extracted UAVid/LoveDA; never download datasets."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Class order is shared with UAVidDataset and exported prediction masks.
UAVID_COLORS = (
    (128, 0, 0), (128, 64, 128), (0, 128, 0), (128, 128, 0),
    (64, 0, 128), (192, 0, 192), (64, 64, 0), (0, 0, 0),
)
# Keep preparation usable without importing PyTorch/OpenMMLab.
OFFICIAL_COUNTS = {
    'loveda': {'train': 2522, 'val': 1669, 'test': 1796},
    'uavid': {'train': 200, 'val': 70, 'test': 150},
}


def inventory(dataset, source):
    """Validate official source splits before writing a single prepared image."""
    splits = {}
    sequences = {}
    for split, expected in OFFICIAL_COUNTS[dataset].items():
        directory = source / (f'uavid_{split}' if dataset == 'uavid' else split.capitalize())
        pattern = 'seq*/Images/*.png' if dataset == 'uavid' else '*/images_png/*.png'
        paths = sorted(directory.glob(pattern))
        if len(paths) != expected:
            raise ValueError(f'{dataset} official {split}: expected {expected} images, got {len(paths)} in {directory}')
        if dataset == 'uavid':
            groups = {}
            for path in paths:
                groups.setdefault(path.parent.parent.name, []).append(path)
            if any(len(frames) != 10 for frames in groups.values()):
                raise ValueError(f'UAVid2020 requires ten annotated frames per sequence: {split}')
            sequences[split] = set(groups)
        else:
            if {p.parent.parent.name for p in paths} != {'Urban', 'Rural'}:
                raise ValueError(f'LoveDA {split} must contain official Urban and Rural domains')
            if len({p.name for p in paths}) != expected:
                raise ValueError(f'Duplicate LoveDA image IDs within {split}')
        if split != 'test':
            label_dir = 'Labels' if dataset == 'uavid' else 'masks_png'
            for image in paths:
                label = image.parent.parent / label_dir / image.name
                if not label.is_file():
                    raise FileNotFoundError(label)
            pattern = 'seq*/Labels/*.png' if dataset == 'uavid' else '*/masks_png/*.png'
            if len(list(directory.glob(pattern))) != expected:
                raise ValueError(f'{dataset} {split} image/label inventory differs')
        splits[split] = paths
    if dataset == 'uavid':
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            overlap = sequences[left] & sequences[right]
            if overlap:
                raise ValueError(f'UAVid sequence leakage between {left}/{right}: {sorted(overlap)}')
    else:
        names = {split: {p.name for p in paths} for split, paths in splits.items()}
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            if names[left] & names[right]:
                raise ValueError(f'LoveDA image IDs overlap between {left}/{right}')
    return splits


def uavid_label_ids(image):
    rgb = np.asarray(image.convert('RGB'))
    labels = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    known = np.all(rgb == (255, 255, 255), axis=-1)
    for index, color in enumerate(UAVID_COLORS):
        matches = np.all(rgb == color, axis=-1)
        labels[matches] = index
        known |= matches
    if not known.all():
        colors = np.unique(rgb[~known].reshape(-1, 3), axis=0)
        raise ValueError(f'Unknown UAVid label colors: {colors[:10].tolist()}')
    return Image.fromarray(labels)


def _save(image, mask, root, split, name):
    image.save(root / 'img_dir' / split / name)
    if mask is not None:
        mask.save(root / 'ann_dir' / split / name)


def _save_tiles(image, mask, root, split, stem, size=1024):
    count = 0
    for y in range(0, image.height, size):
        for x in range(0, image.width, size):
            box = (x, y, min(x + size, image.width), min(y + size, image.height))
            patch = Image.new('RGB', (size, size), 0)
            patch.paste(image.crop(box))
            target = Image.new('L', (size, size), 255)
            target.paste(mask.crop(box))
            _save(patch, target, root, split, f'{stem}_x{x:04d}_y{y:04d}.png')
            count += 1
    return count


def prepare(dataset, source, output):
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f'Refusing to replace prepared data: {output}')
    splits = inventory(dataset, source)
    stage = output.with_name(output.name + '.building')
    if stage.exists():
        raise FileExistsError(f'Incomplete preparation exists: {stage}')
    for split in ('train', 'val', 'test'):
        (stage / 'img_dir' / split).mkdir(parents=True)
        if split != 'test':
            (stage / 'ann_dir' / split).mkdir(parents=True)
    counts = {}
    for split in ('train', 'val', 'test'):
        paths = splits[split]
        count = 0
        names = set()
        for path in paths:
            group = path.parent.parent.name
            stem = f'{group}_{path.stem}'  # retain sequence/domain to avoid collisions
            if stem in names:
                raise ValueError(f'Duplicate image identifier: {stem}')
            names.add(stem)
            with Image.open(path) as stream:
                image = stream.convert('RGB')
            mask = None
            if split != 'test':
                label_path = path.parent.parent / ('Labels' if dataset == 'uavid' else 'masks_png') / path.name
                with Image.open(label_path) as stream:
                    if dataset == 'uavid':
                        mask = uavid_label_ids(stream)
                    else:
                        values = np.asarray(stream)
                        if values.ndim != 2 or not set(np.unique(values)).issubset(set(range(8)) | {255}):
                            raise ValueError(f'Invalid LoveDA IDs: {label_path}')
                        mask = Image.fromarray(values.astype(np.uint8))
                if image.size != mask.size:
                    raise ValueError(f'Image/mask size mismatch: {path}')
            if dataset == 'loveda' and image.size != (1024, 1024):
                raise ValueError(f'LoveDA must be 1024x1024: {path} = {image.size}')
            if dataset == 'uavid' and split == 'train':
                count += _save_tiles(image, mask, stage, split, stem)
            else:
                # Preserve entire UAVid val/test frames, including bottom/right borders.
                _save(image, mask, stage, split, f'{stem}.png')
                count += 1
        counts[split] = dict(source_images=len(paths), prepared_images=count)
    manifest = dict(dataset=dataset, patch_size=1024, counts=counts,
                    official_source_counts=OFFICIAL_COUNTS[dataset],
                    source_files={split: [str(p.relative_to(source)) for p in paths]
                                  for split, paths in splits.items()},
                    sequence_splits={split: sorted({p.parent.parent.name for p in paths})
                                     for split, paths in splits.items()} if dataset == 'uavid' else None,
                    test_labels='private', local_evaluation_split='val',
                    label_encoding='0..7; ignore=255' if dataset == 'uavid' else 'source 1..7; 0 ignored by LoveDADataset')
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('uavid', 'loveda'))
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
