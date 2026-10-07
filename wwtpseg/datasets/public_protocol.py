"""Official split constraints, shared by preparation and production entry points."""
import json
from pathlib import Path

OFFICIAL_COUNTS = {
    'loveda': {'train': 2522, 'val': 1669, 'test': 1796},
    'uavid': {'train': 200, 'val': 70, 'test': 150},
}


def validate_prepared(dataset, root):
    root = Path(root)
    path = root / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'{path}: run tools/remote_sensing/prepare.py on official extracted splits first')
    manifest = json.loads(path.read_text())
    if manifest.get('dataset') != dataset or manifest.get('patch_size') != 1024:
        raise ValueError('Prepared dataset identity or patch size differs from the experiment')
    if manifest.get('official_source_counts') != OFFICIAL_COUNTS[dataset]:
        raise ValueError(f'{dataset} source counts do not match official splits')
    source_files = manifest['source_files']
    seen_source_ids = set()
    for split, expected in OFFICIAL_COUNTS[dataset].items():
        files = source_files[split]
        if len(files) != expected or len(set(files)) != expected:
            raise ValueError(f'{dataset} {split} source file inventory differs')
        info = manifest['counts'][split]
        if info['source_images'] != expected:
            raise ValueError(f'{dataset} {split} source count differs')
        images = sorted((root / 'img_dir' / split).glob('*.png'))
        if len(images) != info['prepared_images'] or not images:
            raise ValueError(f'{dataset} {split} prepared image count differs')
        if dataset == 'loveda' or split != 'train':
            if len(images) != expected:
                raise ValueError(f'{dataset} {split} must retain official frame/image count')
        expected_stems = set()
        sequence_frames = {}
        for filename in files:
            source = Path(filename)
            expected_folder = f'uavid_{split}' if dataset == 'uavid' else split.capitalize()
            expected_leaf = 'Images' if dataset == 'uavid' else 'images_png'
            if len(source.parts) != 4 or source.parts[0] != expected_folder or source.parts[2] != expected_leaf:
                raise ValueError(f'{dataset} {split} source directory differs from the official split')
            group = source.parts[1]
            if dataset == 'loveda' and group not in ('Urban', 'Rural'):
                raise ValueError(f'Unknown LoveDA domain: {group}')
            identifier = (group, source.stem) if dataset == 'uavid' else source.stem
            if identifier in seen_source_ids:
                raise ValueError(f'{dataset} source image repeats across splits')
            seen_source_ids.add(identifier)
            expected_stems.add(f'{group}_{source.stem}')
            sequence_frames[group] = sequence_frames.get(group, 0) + 1
        if dataset == 'uavid' and any(n != 10 for n in sequence_frames.values()):
            raise ValueError(f'UAVid {split} must retain ten frames per sequence')
        actual_stems = {p.stem.rsplit('_x', 1)[0] if dataset == 'uavid' and split == 'train'
                        else p.stem for p in images}
        if actual_stems != expected_stems:
            raise ValueError(f'{dataset} {split} prepared images do not match source frame identities')
        if split != 'test':
            masks = sorted((root / 'ann_dir' / split).glob('*.png'))
            if {p.name for p in masks} != {p.name for p in images}:
                raise ValueError(f'{dataset} {split} image/mask names differ')
    if dataset == 'uavid':
        groups = manifest['sequence_splits']
        for split in ('train', 'val', 'test'):
            observed = {Path(p).parts[1] for p in source_files[split]}
            if observed != set(groups[split]) or len(observed) * 10 != OFFICIAL_COUNTS[dataset][split]:
                raise ValueError(f'UAVid {split} sequence inventory differs')
            if any(p.stem.split('_', 1)[0] not in observed
                   for p in (root / 'img_dir' / split).glob('*.png')):
                raise ValueError(f'UAVid {split} contains a frame from another sequence')
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            if set(groups[left]) & set(groups[right]):
                raise ValueError(f'UAVid sequence leakage between {left}/{right}')
    return manifest
