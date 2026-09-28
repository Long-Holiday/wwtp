#!/usr/bin/env python3
"""Create a v3 initialization from a v2 RGB checkpoint, with strict coverage."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401


def initialize_from_rgb(model, source):
    """Copy every retained tensor; reject v1 / incompatible v2 predictions."""
    source = {key.removeprefix('module.'): value for key, value in source.items()}
    state = model.state_dict()
    required = {key for key in state if not key.startswith('geometry_adapter.')
                and key != 'rgb_anchor_initialized'}
    missing = required - source.keys()
    mismatch = [key for key in required & source.keys() if source[key].shape != state[key].shape]
    if missing or mismatch:
        raise ValueError(f'incompatible RGB anchor: missing={sorted(missing)}, shapes={mismatch}')
    model.load_state_dict({key: source[key] for key in required}, strict=False)
    model.rgb_anchor_initialized.fill_(True)
    return sorted(required)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rgb_checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a fresh initialization path')
    register_all_modules(init_default_scope=True)
    torch.manual_seed(args.seed)
    cfg = Config.fromfile('configs/v3/rpgv_v3_adapter.py')
    cfg.model.rgb_encoder.init_cfg = None
    model = MODELS.build(cfg.model)
    source = torch.load(args.rgb_checkpoint, map_location='cpu', weights_only=False)
    keys = initialize_from_rgb(model, source['state_dict'])
    metadata = dict(architecture='RPGVNetV3', rgb_anchor=str(args.rgb_checkpoint.resolve()),
                    rgb_anchor_sha256=sha256(args.rgb_checkpoint), transferred_keys=keys,
                    seed=args.seed, frozen_rgb=True, optimizer_transferred=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), meta=metadata), args.output)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Initialized {len(keys)} frozen RGB tensors: {args.output}')
    print('Trainable adapter parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == '__main__':
    main()
