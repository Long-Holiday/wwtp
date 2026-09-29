#!/usr/bin/env python3
"""Strictly initialize an RPGV v5.1 warm start from a complete v5 checkpoint.

Run in the project Docker image. This transfers model tensors only; optimizer,
scheduler, epoch, and iteration state are deliberately excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401  Register project models.


ROOT = Path(__file__).resolve().parents[1]
NEW_PREFIXES = ('decoder.spatial8.', 'decoder.spatial4.')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def transfer_v5_state(model: torch.nn.Module, source_state: Mapping) -> tuple[list[str], list[str]]:
    """Require exact old-key coverage and strict-load a complete v5.1 state."""
    target = model.state_dict()
    if not isinstance(source_state, Mapping):
        raise TypeError('source checkpoint has no state_dict mapping')
    if any(not isinstance(key, str) for key in source_state):
        raise ValueError('source state_dict contains a non-string key')
    old_keys = {key for key in target if not key.startswith(NEW_PREFIXES)}
    new_keys = set(target) - old_keys
    if any(not key.startswith(NEW_PREFIXES) for key in new_keys):
        raise AssertionError('new v5.1 keys escaped the allowed prefixes')
    provided = set(source_state)
    missing = sorted(old_keys - provided)
    unexpected = sorted(provided - old_keys)
    if missing or unexpected:
        raise ValueError(f'incompatible v5 key set: missing_old={missing}, '
                         f'unexpected_source={unexpected}')
    for key in sorted(old_keys):
        tensor = source_state[key]
        reference = target[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'source value is not a tensor: {key}')
        if tensor.shape != reference.shape:
            raise ValueError(f'shape mismatch for {key}: '
                             f'{tuple(tensor.shape)} != {tuple(reference.shape)}')
        if tensor.dtype != reference.dtype:
            raise ValueError(f'dtype mismatch for {key}: {tensor.dtype} != {reference.dtype}')
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f'non-finite source tensor: {key}')
    for key in sorted(new_keys):
        if not torch.isfinite(target[key]).all().item():
            raise ValueError(f'non-finite new tensor: {key}')
        if key in ('decoder.spatial8.output.weight', 'decoder.spatial8.output.bias',
                   'decoder.spatial4.output.weight', 'decoder.spatial4.output.bias'):
            if torch.count_nonzero(target[key]).item():
                raise ValueError(f'new spatial output projection is not zero: {key}')
    merged = dict(target)
    merged.update(source_state)
    model.load_state_dict(merged, strict=True)
    return sorted(old_keys), sorted(new_keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='validated RPGV v5 checkpoint')
    parser.add_argument('destination', type=Path, help='new v5.1 initialization checkpoint')
    parser.add_argument('--config', type=Path, default=Path('configs/v51/rpgv_v51.py'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    metadata_path = args.destination.with_suffix('.json')
    if args.destination.exists() or metadata_path.exists():
        parser.error('destination checkpoint or metadata already exists')
    if not args.source.is_file():
        parser.error(f'source checkpoint does not exist: {args.source}')
    if not args.config.is_file():
        parser.error(f'config does not exist: {args.config}')
    if args.source.resolve() == args.destination.resolve():
        parser.error('source and destination must differ')

    cfg = Config.fromfile(str(args.config))
    if cfg.model.get('type') != 'RPGVNetV51':
        parser.error('config must build RPGVNetV51')
    if not cfg.get('required_previous_stage'):
        parser.error('config is not a warm-start v5.1 variant; train from_scratch directly')
    register_all_modules(init_default_scope=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    cfg.model.rgb_encoder.init_cfg = None  # No download or accidental ImageNet reload.
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, Mapping) or 'state_dict' not in checkpoint:
        raise TypeError('source must be a full checkpoint with state_dict')
    old_keys, new_keys = transfer_v5_state(model, checkpoint['state_dict'])
    metadata = {
        'schema': 1,
        'architecture': 'RPGVNetV51',
        'source': str(args.source.resolve()),
        'source_sha256': sha256(args.source),
        'target_config': str(args.config.resolve()),
        'target_config_sha256': sha256(args.config),
        'resolved_config_sha256': hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
        'source_iteration': checkpoint.get('meta', {}).get('iter'),
        'seed': args.seed,
        'transferred_keys': old_keys,
        'new_keys': new_keys,
        'transferred_key_count': len(old_keys),
        'new_key_count': len(new_keys),
        'optimizer_transferred': False,
        'scheduler_transferred': False,
        'iteration_transferred': False,
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open('xb') as stream:
        torch.save({'state_dict': model.state_dict(), 'meta': metadata}, stream)
    with metadata_path.open('x') as stream:
        json.dump(metadata, stream, indent=2)
        stream.write('\n')
    print(f'Initialized {args.destination}: {len(old_keys)} transferred keys, '
          f'{len(new_keys)} new spatial keys; metadata {metadata_path}')


if __name__ == '__main__':
    main()
