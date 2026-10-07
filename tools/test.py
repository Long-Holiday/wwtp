#!/usr/bin/env python3
"""Evaluate the exact model/config used for training."""
import argparse
import json
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from wwtpseg.engine.runtime import prepare_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--work-dir')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg.load_from = str(checkpoint)
    cfg.work_dir = args.work_dir or str(checkpoint.parent / f'eval_{args.split}')
    if args.split == 'val':
        cfg.test_dataloader = cfg.val_dataloader
        cfg.test_evaluator = cfg.val_evaluator
    prepare_runtime(cfg, inference=True)
    metrics = Runner.from_cfg(cfg).test()
    output = Path(cfg.work_dir) / 'metrics.json'
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n')
    print(f'Metrics saved to {output}')


if __name__ == '__main__':
    main()
