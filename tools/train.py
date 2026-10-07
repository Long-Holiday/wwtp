#!/usr/bin/env python3
"""Train any configured WWTP/UAVid/LoveDA experiment with MMEngine."""
import argparse
import os

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from wwtpseg.engine.runtime import prepare_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--work-dir')
    parser.add_argument('--resume', nargs='?', const='auto')
    parser.add_argument('--launcher', choices=('none', 'pytorch', 'slurm', 'mpi'), default='none')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument('--disable-early-stopping', action='store_true')
    parser.add_argument('--no-pretrained', action='store_true', help='offline/debug initialization')
    parser.add_argument('--amp-dtype', choices=('float16', 'bfloat16'), help='dtype of an existing AMP wrapper')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = args.launcher
    cfg.work_dir = args.work_dir or cfg.get('work_dir') or f'work_dirs/{os.path.basename(args.config)[:-3]}'
    if args.resume:
        cfg.resume = True
        cfg.load_from = None if args.resume == 'auto' else args.resume
    prerequisite = cfg.pop('required_previous_stage', None)
    if prerequisite and not cfg.get('load_from') and not cfg.get('resume'):
        raise RuntimeError(f'{prerequisite} is required. Initialize the stage or use --resume.')
    if args.disable_early_stopping:
        cfg.custom_hooks = [hook for hook in cfg.get('custom_hooks', [])
                            if hook.get('type') not in ('EarlyStoppingHook', 'RPGVEarlyStoppingHook')]
    # Early stopping is explicit in configs; never inject a binary metric into public experiments.
    cfg.pop('enable_early_stopping', None)
    if args.amp_dtype:
        if cfg.optim_wrapper.type != 'AmpOptimWrapper':
            raise ValueError('--amp-dtype requires AmpOptimWrapper; FP32 recipes stay FP32.')
        if args.amp_dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('This GPU does not support bfloat16')
        cfg.optim_wrapper.dtype = args.amp_dtype
        if args.amp_dtype == 'bfloat16':
            cfg.optim_wrapper.loss_scale = 1.0
    prepare_runtime(cfg, no_pretrained=args.no_pretrained)
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
