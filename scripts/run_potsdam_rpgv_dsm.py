#!/usr/bin/env python3
"""Separate prepare/train/eval commands for the isolated Potsdam DSM experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / 'data/remote_sensing/prepared/potsdam'
DSM_ROOT = DATA_ROOT / 'dsm_geometry'
CONFIG = 'configs/remote_sensing/potsdam_rpgv_full_dsm.py'
WORK_DIR = REPO_ROOT / 'work_dirs/remote_sensing/potsdam_rpgv_full_dsm'
EVAL_DIR = REPO_ROOT / 'work_dirs/remote_sensing/potsdam_rpgv_full_dsm_eval'


def check_prepared() -> dict[str, int]:
    prepared = json.loads((DATA_ROOT / 'manifest.json').read_text())
    dsm = json.loads((DSM_ROOT / 'manifest.json').read_text())
    size = int(prepared['patch_size'])
    if dsm['patch_size'] != size or dsm['variant'] != 'normalized_lastools.jpg':
        raise ValueError('DSM manifest does not match the prepared Potsdam data')
    expected = math.ceil(6000 / size) ** 2
    groups = {
        'train': prepared['train_tiles'],
        'val': prepared['internal_val_tiles'],
        'test': prepared['official_test_tiles'],
    }
    counts = {}
    for split, tiles in groups.items():
        images = {path.stem for path in (DATA_ROOT / 'img_dir' / split).glob('*.png')}
        geometry = {path.stem for path in (DSM_ROOT / split).glob('*.npz')}
        if len(images) != len(tiles) * expected or geometry != images:
            raise ValueError(
                f'{split}: expected {len(tiles) * expected} RGB/DSM pairs, '
                f'found RGB={len(images)} DSM={len(geometry)}')
        counts[split] = len(images)
    if dsm['counts'] != counts:
        raise ValueError(f'DSM manifest count mismatch: {dsm["counts"]} != {counts}')
    return counts


def docker_command(*args: str) -> list[str]:
    return [
        'docker', 'compose', 'run', '--rm', '--no-deps',
        '-e', 'PYTHONUNBUFFERED=1', 'wwtp', 'python', *args,
    ]


def run(command: list[str], dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('status', 'prepare', 'train', 'eval'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--checkpoint', type=Path,
                        default=WORK_DIR / 'iter_40000.pth',
                        help='eval only; defaults to the 40k DSM checkpoint')
    args = parser.parse_args()

    if args.mode == 'prepare':
        if DSM_ROOT.exists() and not args.dry_run:
            parser.error(f'DSM output already exists: {DSM_ROOT}. '
                         'Run prepare_potsdam_dsm.py --resume explicitly to continue it.')
        run(docker_command('tools/remote_sensing/prepare_potsdam_dsm.py'), args.dry_run)
        return

    try:
        counts = check_prepared()
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.error(f'DSM data is not ready: {error}')
    print(f'DSM patches ready: {counts}')
    if args.mode == 'status':
        print(f'Train checkpoint: {WORK_DIR / "iter_40000.pth"} '
              f'({"present" if (WORK_DIR / "iter_40000.pth").is_file() else "absent"})')
        print(f'Evaluation: {EVAL_DIR / "metrics.json"} '
              f'({"present" if (EVAL_DIR / "metrics.json").is_file() else "absent"})')
        return

    if args.mode == 'train':
        if WORK_DIR.exists() and any(WORK_DIR.iterdir()):
            parser.error(f'DSM training output already exists: {WORK_DIR}')
        command = docker_command(
            'tools/remote_sensing/run.py', 'train', CONFIG,
            '--work-dir', str(WORK_DIR.relative_to(REPO_ROOT)))
    else:
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.is_file():
            parser.error(f'Checkpoint missing: {checkpoint}')
        if not checkpoint.is_relative_to(REPO_ROOT):
            parser.error(f'Checkpoint must be inside the mounted repository: {checkpoint}')
        if EVAL_DIR.exists() and any(EVAL_DIR.iterdir()):
            parser.error(f'DSM evaluation output already exists: {EVAL_DIR}')
        command = docker_command(
            'tools/remote_sensing/run.py', 'eval', CONFIG,
            '--checkpoint', str(checkpoint.relative_to(REPO_ROOT)),
            '--work-dir', str(EVAL_DIR.relative_to(REPO_ROOT)))
    run(command, args.dry_run)


if __name__ == '__main__':
    main()
