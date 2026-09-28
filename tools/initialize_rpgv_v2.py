#!/usr/bin/env python3
"""Explicitly transfer upstream v1 weights; initialize every v2 head afresh."""
import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401


TRANSFER_PREFIXES = (
    'rgb_encoder.', 'global_film.', 'rgb_aux_head.', 'rgb_boundary_head.',
    'rectifier.', 'geometry_encoder.', 'geometry_pyramid.', 'geometry_aux_head.',
    'frequency_validator.', 'high_fusion.', 'low_fusion.',
)


def transfer_upstream(model, source):
    state = model.state_dict()
    transfer = {}
    for key, value in source.items():
        key = key.removeprefix('module.')
        if key.startswith(TRANSFER_PREFIXES):
            if key not in state or value.shape != state[key].shape:
                raise ValueError(f'incompatible upstream tensor: {key}')
            transfer[key] = value
    expected = {key for key in state if key.startswith(TRANSFER_PREFIXES)}
    missing = expected - transfer.keys()
    if missing:
        raise ValueError(f'incomplete upstream checkpoint: {sorted(missing)}')
    model.load_state_dict(transfer, strict=False)
    return sorted(transfer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new checkpoint path')
    register_all_modules(init_default_scope=True)
    torch.manual_seed(args.seed)
    config = Config.fromfile('configs/experiments/rpgv_v2_stage1_rgb.py')
    config.model.rgb_encoder.init_cfg = None
    model = MODELS.build(config.model)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    keys = transfer_upstream(model, source.get('state_dict', source))
    meta = dict(architecture='RPGVNetV2', source=str(args.checkpoint),
                seed=args.seed, transferred_keys=keys,
                new_modules=['decoder', 'refiner'],
                note='Initialization only. No optimizer or training progress is transferred.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), meta=meta), args.output)
    args.output.with_suffix('.transfer.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Transferred {len(keys)} upstream tensors; v2 heads require training: {args.output}')


if __name__ == '__main__':
    main()
