"""Dependency-free source identities and local split rules for DeepGlobe."""
import math
from pathlib import PurePosixPath
import random

SPLIT_PROTOCOL = 'deepglobe_labeled_train_random_v1'
DEFAULT_SPLIT_SEED = 42
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_ROUNDING = 'floor_val_floor_test_remainder_train'
OFFICIAL_FOLDERS = {'official_val': 'valid', 'official_test': 'test'}


def source_id(filename, folder='train'):
    path = PurePosixPath(filename)
    if (len(path.parts) != 2 or path.parts[0] != folder
            or path.as_posix() != filename or not path.name.endswith('_sat.jpg')):
        raise ValueError(f'Invalid DeepGlobe source path: {filename}')
    identifier = path.name.removesuffix('_sat.jpg')
    if not identifier or identifier in ('.', '..'):
        raise ValueError(f'Invalid DeepGlobe source ID: {filename}')
    return identifier


def local_splits(files, seed=DEFAULT_SPLIT_SEED, ratios=DEFAULT_SPLIT_RATIOS):
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('DeepGlobe split seed must be an integer')
    if (len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios)
            or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9)):
        raise ValueError('DeepGlobe split ratios must be three positive values summing to one')
    ordered = sorted(files, key=source_id)
    ids = [source_id(f) for f in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate DeepGlobe labeled source IDs')
    n_val, n_test = math.floor(len(files) * ratios[1]), math.floor(len(files) * ratios[2])
    n_train = len(files) - n_val - n_test
    if min(n_train, n_val, n_test) < 1:
        raise ValueError('DeepGlobe local train/val/test must all be nonempty')
    random.Random(seed).shuffle(ordered)
    return dict(train=sorted(ordered[:n_train]),
                val=sorted(ordered[n_train:n_train + n_val]),
                test=sorted(ordered[n_train + n_val:]))
