#!/usr/bin/env python3
"""Plan, train or evaluate the same seven models across all datasets."""
import argparse
from pathlib import Path
import subprocess
import sys

from wwtpseg.experiments import DATASETS, MAINSTREAM_MODELS, OPTIONAL_MODELS, ROOT, experiment_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'test'))
    parser.add_argument('--dataset', choices=DATASETS, default='wwtp')
    parser.add_argument('--models', nargs='+', choices=MAINSTREAM_MODELS + OPTIONAL_MODELS,
                        default=MAINSTREAM_MODELS)
    parser.add_argument('--work-root', type=Path, default=Path('work_dirs'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpus', type=int, default=1)
    args, extra = parser.parse_known_args()
    if args.gpus < 1:
        parser.error('--gpus must be positive')
    for name in args.models:
        config = str(experiment_config(args.dataset, name).relative_to(ROOT))
        output = args.work_root / args.dataset / name
        if args.mode == 'train':
            command = [sys.executable, 'tools/train.py', config, '--work-dir', str(output)]
            if args.gpus > 1:
                command = [sys.executable, '-m', 'torch.distributed.run', f'--nproc-per-node={args.gpus}',
                           'tools/train.py', config, '--launcher', 'pytorch', '--work-dir', str(output)]
            if args.resume:
                command += ['--resume']
            elif output.exists() and any(output.iterdir()) and not args.dry_run:
                raise FileExistsError(f'Existing experiment: {output}; use --resume or a new --work-root')
        else:
            metric = 'binary_Foreground_IoU' if args.dataset == 'wwtp' else 'mIoU'
            checkpoints = list(output.glob(f'best_{metric}_iter_*.pth'))
            if not checkpoints and not args.dry_run:
                raise FileNotFoundError(f'No validation-selected checkpoint in {output}')
            checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime) if checkpoints else output / f'best_{metric}_iter_SELECTED.pth'
            command = [sys.executable, 'tools/test.py', config, str(checkpoint), '--work-dir', str(output / 'eval')]
        command += extra
        print(' '.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
