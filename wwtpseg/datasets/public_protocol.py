"""Official split constraints, shared by preparation and production entry points."""
import json
from pathlib import Path

from tools.data.deepglobe_protocol import (
    OFFICIAL_FOLDERS, SPLIT_PROTOCOL, SPLIT_ROUNDING, local_splits, source_id)

OFFICIAL_COUNTS = {
    'loveda': {'train': 2522, 'val': 1669, 'test': 1796},
    'uavid': {'train': 200, 'val': 70, 'test': 150},
    'deepglobe': {'train': 6226, 'valid': 1243, 'test': 1101},
}


def _validate_deepglobe(root, manifest):
    metadata = dict(split_protocol=SPLIT_PROTOCOL, split_rounding=SPLIT_ROUNDING,
                    labeled_source_split='train', checkpoint_selection_split='val',
                    test_labels='local_labeled_holdout', local_evaluation_split='test',
                    label_encoding='background=0; road=1; ignore=255')
    if any(manifest.get(key) != value for key, value in metadata.items()):
        raise ValueError('DeepGlobe local split or label protocol differs')
    files = manifest['source_files']
    if set(files) != {'train', 'val', 'test'}:
        raise ValueError('DeepGlobe requires three labeled local splits')
    combined = [filename for split in files.values() for filename in split]
    if len(combined) != OFFICIAL_COUNTS['deepglobe']['train']:
        raise ValueError('DeepGlobe labeled source count differs')
    expected = local_splits(combined, manifest['split_seed'], manifest['split_ratios'])
    if any(sorted(files[split]) != expected[split] for split in expected):
        raise ValueError('DeepGlobe local membership differs from the recorded split algorithm')
    prediction = manifest.get('prediction_source_files', {})
    if not set(prediction).issubset(OFFICIAL_FOLDERS):
        raise ValueError('Unknown DeepGlobe official prediction split')
    if set(manifest['counts']) != set(files) | set(prediction):
        raise ValueError('DeepGlobe prepared split inventory differs')
    seen_ids = {source_id(f) for f in combined}
    for split, filenames in {**files, **prediction}.items():
        folder = 'train' if split in files else OFFICIAL_FOLDERS[split]
        ids = {source_id(f, folder) for f in filenames}
        if len(ids) != len(filenames):
            raise ValueError(f'Duplicate DeepGlobe source IDs: {split}')
        if split in prediction:
            if len(ids) != OFFICIAL_COUNTS['deepglobe'][folder] or seen_ids & ids:
                raise ValueError(f'DeepGlobe official prediction inventory differs: {split}')
            seen_ids.update(ids)
        info = manifest['counts'][split]
        if info != dict(source_images=len(ids), prepared_images=len(ids)):
            raise ValueError(f'DeepGlobe {split} source/prepared counts differ')
        images = {p.name for p in (root / 'img_dir' / split).glob('*.png')}
        if images != {f'{identifier}.png' for identifier in ids}:
            raise ValueError(f'DeepGlobe {split} prepared images differ from source identities')
        masks = {p.name for p in (root / 'ann_dir' / split).glob('*.png')}
        if (split in files and masks != images) or (split in prediction and masks):
            raise ValueError(f'DeepGlobe {split} image/mask inventory differs')
    actual_splits = {p.name for p in (root / 'img_dir').iterdir() if p.is_dir()}
    if actual_splits != set(files) | set(prediction):
        raise ValueError('DeepGlobe image directories differ from manifest splits')


def validate_prepared(dataset, root):
    root = Path(root)
    path = root / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'{path}: run tools/data/prepare_public.py on official extracted splits first')
    manifest = json.loads(path.read_text())
    if manifest.get('dataset') != dataset or manifest.get('patch_size') != 1024:
        raise ValueError('Prepared dataset identity or patch size differs from the experiment')
    if manifest.get('official_source_counts') != OFFICIAL_COUNTS[dataset]:
        raise ValueError(f'{dataset} source counts do not match official splits')
    if dataset == 'deepglobe':
        _validate_deepglobe(root, manifest)
        return manifest
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
