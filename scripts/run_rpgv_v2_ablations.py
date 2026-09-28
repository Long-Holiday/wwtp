#!/usr/bin/env python3
"""Independent single-stage v2 ablations; no staged or cross-variant weights."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from mmengine.config import Config
from wwtpseg.engine.early_stopping import validation_history, plateau_reached

ROOT = Path(__file__).resolve().parents[1]
PAPER = ('full', 'no_rgr', 'no_frequency_validation', 'unweighted_fusion',
         'no_contour', 'no_structure_regularization')
PROGRESSIVE = ('progressive_base', 'progressive_rgr', 'progressive_dfgv',
               'progressive_weighted', 'progressive_contour', 'full')
CORE = PAPER
VARIANTS = tuple(dict.fromkeys((*PROGRESSIVE, *PAPER)))
RUN = 'joint'
BUDGET = 100000
METRIC = 'binary/Foreground_IoU'


def read_validation(work_dir):
    """Only finite training validations, deduplicated across resumes."""
    return validation_history(work_dir, METRIC)


def select_best(work_dir, budget, allow_early_stop=False):
    rows = read_validation(work_dir)
    last = max((r['selected_iter'] for r in rows), default=0)
    if not rows or last > budget or (not allow_early_stop and last != budget):
        raise RuntimeError(f'{work_dir}: no valid final validation within {budget}')
    # CheckpointHook uses strict greater-than: the first maximum wins ties.
    best = max(sorted(rows, key=lambda r: r['selected_iter']), key=lambda r: r[METRIC])
    path = Path(work_dir) / f"best_binary_Foreground_IoU_iter_{best['selected_iter']}.pth"
    if not path.is_file():
        raise FileNotFoundError(f'validation-selected checkpoint is missing: {path}')
    return path.resolve(), best


def training_result(work_dir, budget, early_stopping=None):
    rows = read_validation(work_dir)
    last = max((r['selected_iter'] for r in rows), default=0)
    stopped_early = 0 < last < budget
    if stopped_early and (not early_stopping or not plateau_reached(rows, early_stopping)):
        raise RuntimeError(f'{work_dir}: incomplete run, early-stop patience was not reached')
    best, metrics = select_best(work_dir, budget, allow_early_stop=stopped_early)
    return best, metrics, last, stopped_early


def completed_result(work_dir, budget, early_stopping=None):
    record = json.loads((work_dir / 'completed.json').read_text())
    best, metrics, last, stopped = training_result(work_dir, budget, early_stopping)
    if (record['initialization'] is not None or record['budget'] != budget
            or record['last_val_iter'] != last or record['stopped_early'] != stopped
            or file_hash(best) != record['checkpoint_sha256']):
        raise RuntimeError(f'{work_dir}: completed run changed')
    return best, metrics


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def make_config(variant, seed, work_dir, max_iters=BUDGET):
    if max_iters < 2000 or max_iters % 2000:
        raise ValueError('max_iters must be a positive multiple of 2000')
    cfg = Config.fromfile(ROOT / f'configs/ablations_v2/{variant}.py')
    cfg.randomness = dict(seed=seed, deterministic=True)
    cfg.env_cfg.cudnn_benchmark = False
    cfg.load_from = None
    cfg.resume = False
    cfg.work_dir = str(work_dir)
    cfg.train_cfg.max_iters = max_iters
    cfg.param_scheduler[-1].end = max_iters
    assert cfg.model.training_stage == 'joint'
    assert not cfg.get('required_previous_stage')
    assert cfg.enable_early_stopping
    assert not cfg.model.learnable_loss_weights
    return cfg


def code_signature():
    paths = sorted((ROOT / 'wwtpseg').rglob('*.py')) + [
        Path(__file__), ROOT / 'tools/train.py', ROOT / 'tools/evaluate_rpgv_v2_ablation.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def prepare(root, protocol, variant, seed, max_iters=BUDGET):
    if protocol != 'single_stage':
        raise ValueError('Only single_stage is supported; use a fresh work root')
    directory = Path(root).resolve() / protocol / f'seed_{seed}' / variant
    cfg = make_config(variant, seed, directory / RUN, max_iters)
    content = cfg.pretty_text
    identity = dict(schema=3, protocol=protocol, variant=variant, seed=seed,
                    budgets={RUN: max_iters}, initialization=None,
                    loss_weighting='fixed', early_stopping=dict(cfg.custom_hooks[0]),
                    configs={RUN: hashlib.sha256(content.encode()).hexdigest()},
                    code=code_signature())
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'experiment.json'
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        if existing != identity:
            raise RuntimeError(f'plan/config/code changed: use a fresh work root instead of {directory}')
    else:
        atomic_json(manifest, identity)
    config_dir = directory / 'configs'
    config_dir.mkdir(exist_ok=True)
    config = config_dir / f'{RUN}.py'
    if config.exists() and file_hash(config) != identity['configs'][RUN]:
        raise RuntimeError(f'planned config changed: {config}')
    if not config.exists():
        config.write_text(content)
    return directory, identity


def verify_plan(directory, identity):
    config = directory / 'configs' / f'{RUN}.py'
    if file_hash(config) != identity['configs'][RUN]:
        raise RuntimeError(f'planned config changed: {config}')
    cfg = Config.fromfile(config)
    if (cfg.get('load_from') or cfg.get('resume') or cfg.get('required_previous_stage')
            or cfg.model.training_stage != 'joint' or identity['initialization'] is not None):
        raise RuntimeError('single-stage runs cannot initialize from staged/variant checkpoints')
    if identity.get('schema', 1) >= 3 and (
            cfg.model.learnable_loss_weights or identity.get('loss_weighting') != 'fixed'
            or identity.get('early_stopping') != dict(cfg.custom_hooks[0])):
        raise RuntimeError('fixed loss weights or early-stopping policy changed')


def run_experiment(directory, identity, resume=False):
    verify_plan(directory, identity)
    if code_signature() != identity['code']:
        raise RuntimeError('training code changed since planning; use a fresh work root')
    budget = identity['budgets'][RUN]
    work_dir = directory / RUN
    done = work_dir / 'completed.json'
    if done.exists():
        completed_result(work_dir, budget, identity.get('early_stopping'))
        print(f'[SKIP completed] {work_dir}', flush=True)
        return
    command = [sys.executable, str(ROOT / 'tools/train.py'),
               str(directory / 'configs' / f'{RUN}.py'), '--work-dir', str(work_dir)]
    started = work_dir / 'started.json'
    if work_dir.exists() and any(work_dir.iterdir()):
        if not resume or not started.exists():
            raise RuntimeError(f'{work_dir}: incomplete run; use --resume-incomplete or a fresh root')
        if json.loads(started.read_text())['initialization'] is not None:
            raise RuntimeError(f'{work_dir}: initialization changed during resume')
        pointer = work_dir / 'last_checkpoint'
        checkpoint = Path(pointer.read_text().strip()) if pointer.is_file() else None
        if checkpoint is None or not checkpoint.is_file() or checkpoint.resolve().parent != work_dir.resolve():
            raise RuntimeError(f'{work_dir}: no valid local last_checkpoint to resume')
        command += ['--resume']
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(started, dict(initialization=None))
    print('[TRAIN]', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    best, metrics, last, stopped = training_result(work_dir, budget, identity.get('early_stopping'))
    atomic_json(done, dict(initialization=None,
                           checkpoint=str(best), checkpoint_sha256=file_hash(best),
                           budget=budget, last_val_iter=last,
                           stopped_early=stopped, validation=metrics))


def evaluate(directory, split):
    identity = json.loads((directory / 'experiment.json').read_text())
    verify_plan(directory, identity)
    work_dir = directory / RUN
    done = work_dir / 'completed.json'
    if not done.exists():
        raise RuntimeError(f'{work_dir}: training completion has not been recorded')
    checkpoint, _ = completed_result(work_dir, identity['budgets'][RUN],
                                     identity.get('early_stopping'))
    subprocess.run([sys.executable, str(ROOT / 'tools/evaluate_rpgv_v2_ablation.py'),
                    str(directory / 'configs' / f'{RUN}.py'), str(checkpoint),
                    '--split', split, '--output', str(directory / f'evaluation_{split}')],
                   cwd=ROOT, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run', 'evaluate'])
    parser.add_argument('--protocol', choices=['single_stage'], default='single_stage')
    parser.add_argument('--variants', nargs='+', default=['progressive'],
                        choices=['progressive', 'paper', 'core', 'all', *VARIANTS],
                        help='progressive: cumulative additions; paper/core: one removal at a time; all: both')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--work-root', type=Path, default=Path('work_dirs/rpgv_v2_single_stage_fixed'))
    parser.add_argument('--max-iters', type=int, default=BUDGET)
    parser.add_argument('--resume-incomplete', action='store_true')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    args = parser.parse_args()
    groups = dict(progressive=PROGRESSIVE, paper=PAPER, core=CORE, all=VARIANTS)
    names = [name for item in args.variants for name in groups.get(item, (item,))]
    args.variants = list(dict.fromkeys([*names, 'full']))
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('seeds must be distinct nonnegative integers')
    if args.max_iters < 2000 or args.max_iters % 2000:
        parser.error('--max-iters must be a positive multiple of 2000')
    return args


def main():
    os.chdir(ROOT)
    args = parse_args()
    for seed in args.seeds:
        for variant in args.variants:
            directory = args.work_root.resolve() / args.protocol / f'seed_{seed}' / variant
            if args.action == 'evaluate':
                # Read immutable manifests; evaluation never replans training.
                identity = json.loads((directory / 'experiment.json').read_text())
            else:
                directory, identity = prepare(args.work_root, args.protocol, variant,
                                              seed, args.max_iters)
            print(f'[{args.action}] {args.protocol} seed={seed} {variant}: {identity["budgets"]}', flush=True)
            if args.action == 'plan':
                continue
            with (directory / '.runner.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.action == 'run':
                    run_experiment(directory, identity, args.resume_incomplete)
                else:
                    evaluate(directory, args.split)


if __name__ == '__main__':
    main()
