#!/usr/bin/env python3
"""Run the six full-view Potsdam experiments sequentially and record results."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/remote_sensing/prepared/potsdam'
WORK = ROOT / 'work_dirs/remote_sensing'
NAMES = ('deeplabv3plus', 'mask2former', 'segformer', 'unetformer',
         'rpgv_v2_depth_anything', 'rpgv_v2_ndsm')


def docker(*args, dry_run=False):
    if os.getenv('POTSDAM_IN_CONTAINER') == '1':
        command = list(args)
    else:
        command = ['docker', 'compose', 'run', '--rm', '--no-deps', 'wwtp', *args]
    print(' '.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def gpu_busy():
    # Total GPU memory is visible across containers. Per-process queries can
    # hide processes in another PID namespace and wrongly report an idle GPU.
    query = ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits']
    result = subprocess.run(query, text=True, capture_output=True)
    if result.returncode:
        result = subprocess.run(
            ['docker', 'run', '--rm', '--gpus', 'all', '--entrypoint',
             'nvidia-smi', 'wwtp-mmseg:1.2.2-cu121', *query[1:]],
            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError('Cannot verify GPU idleness with nvidia-smi')
    try:
        used_mib = [int(line.strip()) for line in result.stdout.splitlines()
                    if line.strip()]
    except ValueError as exc:
        raise RuntimeError(f'Unexpected nvidia-smi output: {result.stdout}') from exc
    if not used_mib:
        raise RuntimeError('nvidia-smi returned no GPUs')
    return any(used > 512 for used in used_mib)


def wait_gpu():
    idle_checks = 0
    while idle_checks < 2:
        if gpu_busy():
            idle_checks = 0
            print('GPU memory is in use; waiting 60 seconds', flush=True)
        else:
            idle_checks += 1
            if idle_checks < 2:
                print('GPU appears idle; confirming in 60 seconds', flush=True)
        time.sleep(60)


def config(name):
    if name not in NAMES:
        raise ValueError(name)
    return f'configs/remote_sensing/potsdam_original_{name}.py'


def prepare_depth(dry_run=False):
    docker('python', 'tools/remote_sensing/generate_potsdam_full_depth.py',
           dry_run=dry_run)


def prepare_ndsm(dry_run=False):
    docker('python', 'tools/remote_sensing/prepare_potsdam_full_ndsm.py',
           dry_run=dry_run)


def train(name, dry_run=False):
    work = WORK / f'potsdam_original_{name}'
    best = sorted(work.glob('best_mIoU*.pth'), key=lambda p: p.stat().st_mtime)
    if best and not dry_run:
        print(f'{name}: best checkpoint already exists: {best[-1]}', flush=True)
        return best[-1]
    if work.exists() and any(work.iterdir()) and not dry_run:
        raise FileExistsError(f'{work} has no best checkpoint; inspect it before resuming')
    docker('python', 'tools/remote_sensing/run.py', 'train', config(name),
           '--work-dir', f'work_dirs/remote_sensing/potsdam_original_{name}',
           dry_run=dry_run)
    if dry_run:
        return None
    best = sorted(work.glob('best_mIoU*.pth'), key=lambda p: p.stat().st_mtime)
    if not best:
        raise FileNotFoundError(f'No validation-selected checkpoint in {work}')
    return best[-1]


def evaluate(name, checkpoint=None, dry_run=False):
    work = WORK / f'potsdam_original_{name}_eval'
    metrics = work / 'metrics.json'
    if metrics.exists() and not dry_run:
        return json.loads(metrics.read_text())
    if checkpoint is None:
        candidates = sorted(
            (WORK / f'potsdam_original_{name}').glob('best_mIoU*.pth'),
            key=lambda p: p.stat().st_mtime)
        if not candidates and not dry_run:
            raise FileNotFoundError(f'No validation-selected checkpoint for {name}')
        checkpoint = candidates[-1] if candidates else Path('best_mIoU_iter_XXXX.pth')
    docker('python', 'tools/remote_sensing/run.py', 'eval', config(name),
           '--checkpoint', f'work_dirs/remote_sensing/potsdam_original_{name}/{checkpoint.name}',
           '--work-dir', f'work_dirs/remote_sensing/potsdam_original_{name}_eval',
           dry_run=dry_run)
    return None if dry_run else json.loads(metrics.read_text())


def status():
    for name in NAMES:
        work = WORK / f'potsdam_original_{name}'
        best = list(work.glob('best_mIoU*.pth'))
        metrics = WORK / f'potsdam_original_{name}_eval' / 'metrics.json'
        print(f'{name}: best={len(best)}, evaluated={metrics.exists()}')
    for name in ('geometry_official_ndsm_1536', 'geometry_depth_anything_1536'):
        path = DATA / name
        print(f'{name}: {len(list(path.glob("*.npz")))} / 37')


def write_summary():
    rows = []
    for name in NAMES:
        path = WORK / f'potsdam_original_{name}_eval' / 'metrics.json'
        if not path.is_file():
            raise FileNotFoundError(path)
        metrics = json.loads(path.read_text())
        rows.append(dict(model=name,
                         mIoU5=metrics['full/mIoU5'],
                         mF15=metrics['full/mF15'],
                         OA5=metrics['full/OA5'], metrics=metrics))
    output = WORK / 'potsdam_original_results.json'
    output.write_text(json.dumps(dict(
        protocol='22 train, 2_10 validation, 7_10 excluded, 14 official test',
        grid='whole 300m tile resized to 1536x1536 (0.1953125 m/px)',
        selection='best validation mIoU checkpoint, fixed 3000 iterations; no early stopping',
        results=rows), indent=2) + '\n')
    print(f'Combined results: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'prepare-depth',
                                           'prepare-ndsm', 'all',
                                           'train', 'eval'))
    parser.add_argument('--model', choices=NAMES)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--wait-gpu', action='store_true')
    args = parser.parse_args()
    if args.action in ('train', 'eval') and not args.model:
        parser.error('--model is required for train or eval')
    if args.action == 'status':
        status()
        return
    if not args.dry_run:
        if args.wait_gpu:
            wait_gpu()
        elif gpu_busy():
            raise RuntimeError('Another GPU process is active; use --wait-gpu to queue')
    if args.action == 'prepare-depth':
        prepare_depth(args.dry_run)
    elif args.action == 'prepare-ndsm':
        prepare_ndsm(args.dry_run)
    elif args.action == 'train':
        train(args.model, args.dry_run)
    elif args.action == 'eval':
        evaluate(args.model, dry_run=args.dry_run)
    else:
        prepare_ndsm(args.dry_run)
        prepare_depth(args.dry_run)
        for name in NAMES:
            if not args.dry_run and args.wait_gpu:
                wait_gpu()
            checkpoint = train(name, args.dry_run)
            evaluate(name, checkpoint, args.dry_run)
        if not args.dry_run:
            write_summary()
        status()


if __name__ == '__main__':
    main()
