#!/usr/bin/env python3
"""Full, matched RGB-anchor vs geometry-adapter evaluation for one v3 checkpoint."""
import argparse
import copy
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner

from initialize_rpgv_v3 import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/v3/rpgv_v3_adapter.py'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    args = parser.parse_args()
    data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not bool(data['state_dict'].get('rgb_anchor_initialized', False)):
        raise ValueError('checkpoint lacks a verified v3 RGB anchor')
    del data
    identity = dict(checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_sha256=sha256(args.checkpoint), config_sha256=sha256(args.config),
                    split=args.split, postprocessing=False)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'identity.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('evaluation identity changed; use a fresh output directory')
    manifest.write_text(json.dumps(identity, indent=2) + '\n')
    scores = {}
    for name, strength in (('rgb_anchor', 0.0), ('with_adapter', 1.0)):
        result = args.output / f'{name}.json'
        if result.exists():
            scores[name] = json.loads(result.read_text())
            continue
        cfg = Config.fromfile(args.config)
        cfg.load_from, cfg.resume = str(args.checkpoint), False
        cfg.model.correction_strength = strength
        cfg.work_dir = str(args.output / name)
        if args.split == 'val':
            cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
            cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
        cfg.test_evaluator.append(dict(type='BinaryTopologyMetric', small_area=256))
        runner = Runner.from_cfg(cfg)
        scores[name] = runner.test()
        result.write_text(json.dumps(scores[name], indent=2) + '\n')
        del runner
    metric = 'binary/Foreground_IoU'
    summary = dict(**identity, metrics=scores,
                   adapter_gain_iou_pp=scores['with_adapter'][metric] - scores['rgb_anchor'][metric])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
