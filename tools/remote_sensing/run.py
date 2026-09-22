#!/usr/bin/env python3
"""Train or evaluate an isolated MMSeg remote-sensing experiment."""

import argparse
import json
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'eval'])
    parser.add_argument('config', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='required for eval')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()

    cfg = Config.fromfile(str(args.config))
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    root = Path(cfg['data_root'])
    if not root.joinpath('manifest.json').is_file():
        raise FileNotFoundError(f'Prepared dataset missing: {root}. Run prepare.py first.')
    if args.mode == 'train':
        work_dir = args.work_dir or Path('work_dirs/remote_sensing') / args.config.stem
        if work_dir.exists() and any(work_dir.iterdir()) and not args.resume:
            raise FileExistsError(f'Experiment output exists; choose a new --work-dir or --resume: {work_dir}')
        cfg.work_dir = str(work_dir)
        cfg.resume = args.resume
        Runner.from_cfg(cfg).train()
    else:
        if not args.checkpoint or not args.checkpoint.is_file():
            parser.error('eval requires an existing --checkpoint')
        work_dir = args.work_dir or Path('work_dirs/remote_sensing') / f'{args.config.stem}_eval'
        work_dir.mkdir(parents=True, exist_ok=True)
        output = work_dir / 'metrics.json'
        if output.exists():
            raise FileExistsError(f'Refusing to replace existing metrics: {output}')
        cfg.work_dir = str(work_dir)
        cfg.load_from = str(args.checkpoint)
        cfg.resume = False
        if cfg.model.get('backbone') and cfg.model.backbone.get('init_cfg'):
            cfg.model.backbone.init_cfg = None  # full experiment checkpoint supplies these weights
        if cfg.model.get('rgb_encoder') and cfg.model.rgb_encoder.get('init_cfg'):
            cfg.model.rgb_encoder.init_cfg = None
        metrics = Runner.from_cfg(cfg).test()
        with output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n')
        print(f'Metrics saved to {output}')


if __name__ == '__main__':
    main()
