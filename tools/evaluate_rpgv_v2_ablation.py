#!/usr/bin/env python3
"""Evaluate one validation-selected WWTP checkpoint, including mask topology."""
import argparse
import copy
import json
from pathlib import Path
import sys

from mmengine.config import Config
from mmengine.runner import Runner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_rpgv_v2_ablations import atomic_json, file_hash


def validate_checkpoint(config_path, checkpoint):
    """Strictly validate inference weights, including learned loss parameters."""
    import torch
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules
    import wwtpseg  # noqa: F401

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(config_path)
    cfg.model.rgb_encoder.init_cfg = None
    model = MODELS.build(cfg.model)
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(source['state_dict'], strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    identity = dict(split=args.split, checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_sha256=file_hash(args.checkpoint),
                    config_sha256=file_hash(args.config), small_area_px=256,
                    connectivity=8, postprocessing=False)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = args.output / 'identity.json'
    if metadata.exists() and json.loads(metadata.read_text()) != identity:
        raise RuntimeError('evaluation identity changed; choose a fresh output directory')
    atomic_json(metadata, identity)
    result = args.output / 'metrics.json'
    if result.exists():
        print(f'[SKIP evaluated] {result}')
        return
    validate_checkpoint(args.config, args.checkpoint)
    cfg = Config.fromfile(args.config)
    cfg.load_from = str(args.checkpoint.resolve())
    cfg.resume = False
    cfg.work_dir = str(args.output.resolve())
    if args.split == 'val':
        cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
        cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
    cfg.test_evaluator.append(dict(type='BinaryTopologyMetric', small_area=256))
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    metrics['complexity/parameters'] = sum(p.numel() for p in model.parameters())
    metrics['complexity/prediction_parameters'] = sum(
        p.numel() for m in (model.decoder, model.refiner, model.detail_refiner)
        for p in m.parameters())
    atomic_json(result, dict(**identity, metrics=metrics))
    print(f'Metrics saved to {result}')


if __name__ == '__main__':
    main()
