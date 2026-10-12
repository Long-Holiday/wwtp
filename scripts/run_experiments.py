#!/usr/bin/env python3
"""Plan, train and evaluate baseline, RPGV v6/v7/v8 and ablation suites."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

# Loading the lightweight roster directly avoids importing the training stack for dry-run.
_roster = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'wwtpseg/experiments/__init__.py'))
DATASETS = _roster['DATASETS']
EVALUATION_PROTOCOLS = _roster['EVALUATION_PROTOCOLS']
MAINSTREAM_MODELS = _roster['MAINSTREAM_MODELS']
OPTIONAL_MODELS = _roster['OPTIONAL_MODELS']
OFFLINE_GEOMETRY_MODELS = _roster['OFFLINE_GEOMETRY_MODELS']
EXPERIMENT_SUITES = _roster['EXPERIMENT_SUITES']
ROOT = _roster['ROOT']
experiment_config = _roster['experiment_config']


def run_command(command, dry_run):
    print(shlex.join(command), flush=True)
    if not dry_run:
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(ROOT), env.get('PYTHONPATH'))))
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def extract_primary_metric(metrics_dict, dataset='wwtp'):
    if not isinstance(metrics_dict, dict):
        return None
    metric = EVALUATION_PROTOCOLS[dataset]['metric']
    keys = (metric, metric.replace('/', '_'), metric.rsplit('/', 1)[-1])
    for key in keys:
        if key in metrics_dict:
            return float(metrics_dict[key])
    return None


def check_v7_superiority(work_root, dataset='wwtp'):
    """Compare RPGV v7 test metric against all mainstream baselines.

    Returns (is_superior, v7_score, best_baseline_name, best_baseline_score, baseline_scores)
    """
    v7_eval = work_root / dataset / 'rpgv_v7' / 'eval' / 'metrics.json'
    if not v7_eval.is_file():
        return None, None, None, None, {}
    try:
        v7_metrics = json.loads(v7_eval.read_text())
    except Exception:
        return None, None, None, None, {}
    v7_score = extract_primary_metric(v7_metrics, dataset)
    if v7_score is None:
        return None, None, None, None, {}

    baseline_scores = {}
    for name in MAINSTREAM_MODELS:
        eval_path = work_root / dataset / name / 'eval' / 'metrics.json'
        if eval_path.is_file():
            try:
                b_metrics = json.loads(eval_path.read_text())
                score = extract_primary_metric(b_metrics, dataset)
                if score is not None:
                    baseline_scores[name] = score
            except Exception:
                pass
    if not baseline_scores:
        return None, v7_score, None, None, {}
    best_baseline = max(baseline_scores, key=baseline_scores.get)
    best_baseline_score = baseline_scores[best_baseline]
    is_superior = v7_score > best_baseline_score
    return is_superior, v7_score, best_baseline, best_baseline_score, baseline_scores


def best_checkpoint(output, dataset, dry_run):
    metric = EVALUATION_PROTOCOLS[dataset]['metric'].replace('/', '_')
    checkpoints = [path for path in output.glob(f'best_{metric}_iter_*.pth') if path.is_file()]
    if checkpoints:
        return max(checkpoints, key=lambda path: path.stat().st_mtime_ns)
    if dry_run:
        return output / f'best_{metric}_iter_SELECTED.pth'
    raise FileNotFoundError(f'No validation-selected checkpoint in {output}')


def geometry_jobs(configs, dataset, splits, cfg_options, dry_run):
    if dry_run:
        if dataset == 'wwtp':
            data_root = os.getenv('WWTP_DATA_ROOT', os.getenv('WWTP_V5_DATA_ROOT', 'wwtp_semantic_dataset_1m'))
            pseudo_root = os.getenv('WWTP_PSEUDO_ROOT', str(Path(data_root) / 'pseudo_geometry'))
        else:
            data_root = os.getenv(f'{dataset.upper()}_DATA_ROOT', f'data/remote_sensing/prepared/{dataset}')
            pseudo_root = os.getenv('PUBLIC_PSEUDO_ROOT', str(Path(data_root) / 'pseudo_geometry'))
        return {(data_root, pseudo_root, 'images' if dataset == 'wwtp' else 'img_dir'): set(splits)}

    from mmengine.config import Config, DictAction

    overrides_parser = argparse.ArgumentParser()
    overrides_parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    overrides = overrides_parser.parse_args(['--cfg-options', *cfg_options]).cfg_options if cfg_options else {}
    jobs = {}
    for config in configs:
        cfg = Config.fromfile(config, import_custom_modules=False)
        cfg.merge_from_dict(overrides)
        for split in splits:
            data = cfg[f'{split}_dataloader'].dataset
            geometry = next((step for step in data.pipeline if step['type'] == 'LoadPseudoGeometry'), None)
            if geometry is None:
                continue
            prefix = Path(data.data_prefix.img_path)
            if prefix.name != split or prefix.parent.as_posix() not in ('images', 'img_dir'):
                raise ValueError(f'Unsupported geometry image layout: {prefix}; prepare geometry manually')
            key = (str(data.data_root), str(geometry['pseudo_root']), prefix.parent.as_posix())
            jobs.setdefault(key, set()).add(split)
    return jobs


def prepare_geometry(configs, dataset, splits, cfg_options, dry_run):
    for (data_root, pseudo_root, image_dir), job_splits in geometry_jobs(
            configs, dataset, splits, cfg_options, dry_run).items():
        missing = dry_run
        for split in sorted(job_splits):
            if dry_run:
                continue
            images_dir = Path(data_root) / image_dir / split
            if not images_dir.is_absolute():
                images_dir = ROOT / images_dir
            images = list(images_dir.glob('*.png'))
            if not images:
                raise FileNotFoundError(f'No PNG images found in {images_dir}')
            cache = Path(pseudo_root) / split
            if not cache.is_absolute():
                cache = ROOT / cache
            missing |= any(not (cache / f'{image.stem}.npz').is_file() for image in images)
        if missing:
            print('Preparing missing pseudo geometry (dry-run paths use environment defaults).' if dry_run
                  else f'Generating missing pseudo geometry in {pseudo_root}', flush=True)
            run_command([sys.executable, 'tools/data/generate_pseudo_geometry.py', data_root,
                         '--output-root', pseudo_root, '--image-dir', image_dir,
                         '--splits', *sorted(job_splits)], dry_run)
        else:
            print(f'Pseudo geometry is complete in {pseudo_root}; skipping generation.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'test', 'train-test'))
    parser.add_argument('--dataset', choices=DATASETS, default='wwtp')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--models', nargs='+', choices=MAINSTREAM_MODELS + OPTIONAL_MODELS)
    selection.add_argument('--suite', choices=EXPERIMENT_SUITES)
    parser.add_argument('--work-root', type=Path, default=Path('work_dirs'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite and clear existing output directory instead of erroring when not resuming')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--split', choices=('val', 'test'),
                        help='Evaluation split; UAVid/LoveDA use val, WWTP/DeepGlobe default to labeled test')
    parser.add_argument('--exclude', nargs='+', default=[], choices=MAINSTREAM_MODELS + OPTIONAL_MODELS,
                        help='Drop models from the selected suite or model list')
    parser.add_argument('--cfg-options', nargs='+', default=[], help='Shared train/evaluation config overrides')
    parser.add_argument('--gate-v7-superiority', action='store_true',
                        default=os.getenv('GATE_V7_SUPERIORITY', '1') in ('1', 'true', 'True'),
                        help='Skip ablation suite if v7 test metric is not strictly higher than all baselines')
    parser.add_argument('--auto-prerequisites', action='store_true',
                        default=os.getenv('AUTO_PREREQUISITES', '1') in ('1', 'true', 'True'),
                        help='Automatically train and evaluate v8 and fresh 30k retrained v7 before running ablations')
    parser.add_argument('--no-auto-prerequisites', action='store_false', dest='auto_prerequisites')
    args, extra = parser.parse_known_args(argv)
    if args.gpus < 1:
        parser.error('--gpus must be positive')
    if not EVALUATION_PROTOCOLS[args.dataset]['labeled_test'] and args.split == 'test':
        parser.error('Public test labels are private; use --split val')
    if args.suite in ('v7-ablations', 'v6-ablations') and not args.dry_run:
        work_root_resolved = args.work_root if args.work_root.is_absolute() else ROOT / args.work_root
        if args.suite == 'v7-ablations' and args.dataset == 'wwtp' and args.auto_prerequisites:
            # 1. Ensure RPGV v8 is trained and evaluated first
            v8_eval = work_root_resolved / 'wwtp' / 'rpgv_v8' / 'eval' / 'metrics.json'
            if not v8_eval.is_file():
                print('[PIPELINE-AUTO] Pre-requisite RPGV v8 evaluation not found. Running RPGV v8 first...', flush=True)
                v8_cmd = [sys.executable, str(Path(__file__).resolve()), 'train-test', '--dataset', 'wwtp',
                          '--models', 'rpgv_v8', '--resume', '--work-root', str(work_root_resolved)]
                if args.cfg_options:
                    v8_cmd += ['--cfg-options', *args.cfg_options]
                run_command(v8_cmd, args.dry_run)

            # 2. Ensure RPGV v7 fresh 30k retraining is completed
            v7_marker = work_root_resolved / 'wwtp' / 'rpgv_v7' / '.retrained_30k_done'
            if not v7_marker.is_file():
                print('[PIPELINE-AUTO] Pre-requisite RPGV v7 30k fresh retraining not marked. Retraining v7...', flush=True)
                v7_cmd = [sys.executable, str(Path(__file__).resolve()), 'train-test', '--dataset', 'wwtp',
                          '--suite', 'v7', '--overwrite', '--work-root', str(work_root_resolved)]
                if args.cfg_options:
                    v7_cmd += ['--cfg-options', *args.cfg_options]
                run_command(v7_cmd, args.dry_run)
                v7_marker.parent.mkdir(parents=True, exist_ok=True)
                v7_marker.write_text('done\n')
                print('[PIPELINE-AUTO] RPGV v7 30k fresh retraining complete.', flush=True)

        if args.gate_v7_superiority:
            is_superior, v7_score, best_baseline, best_score, baseline_scores = check_v7_superiority(
                work_root_resolved, args.dataset)
            if is_superior is False and best_baseline is not None:
                print(f"[GATE] RPGV v7 test metric ({v7_score:.4f}) is NOT higher than best baseline "
                      f"{best_baseline} ({best_score:.4f}). "
                      f"Skipping ablation experiments as requested.", flush=True)
                return
            elif is_superior is True:
                print(f"[GATE] RPGV v7 test metric ({v7_score:.4f}) is higher than best baseline "
                      f"{best_baseline} ({best_score:.4f}). Proceeding with ablations.", flush=True)
    models = args.models or EXPERIMENT_SUITES[args.suite or 'baselines']
    models = [name for name in models if name not in args.exclude]
    try:
        configs = [experiment_config(args.dataset, name) for name in models]
    except ValueError as error:
        parser.error(str(error))
    outputs = [args.work_root / args.dataset / name for name in models]
    outputs = [path if path.is_absolute() else ROOT / path for path in outputs]
    training = args.mode in ('train', 'train-test')
    if training and not args.resume and not args.dry_run:
        for output in outputs:
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                if args.overwrite:
                    import shutil
                    print(f'Overwriting existing experiment directory: {output}', flush=True)
                    shutil.rmtree(output)
                else:
                    raise FileExistsError(f'Existing experiment: {output}; use --resume, --overwrite or a new --work-root')
    for config in configs:
        if not config.is_file():
            raise FileNotFoundError(config)
    split = args.split or EVALUATION_PROTOCOLS[args.dataset]['split']
    shared = ['--cfg-options', *args.cfg_options] if args.cfg_options else []
    geometry_configs = [config for name, config in zip(models, configs) if name in OFFLINE_GEOMETRY_MODELS]
    if geometry_configs:
        splits = list(dict.fromkeys((['train', 'val'] if training else []) +
                                    ([split] if args.mode != 'train' else [])))
        prepare_geometry(geometry_configs, args.dataset, splits, args.cfg_options, args.dry_run)
    for name, config_path, output in zip(models, configs, outputs):
        config = str(config_path.relative_to(ROOT))
        print(f'[{args.dataset}/{name}] {args.mode}', flush=True)
        if training:
            command = [sys.executable, 'tools/train.py', config, '--work-dir', str(output)]
            if args.gpus > 1:
                command = [sys.executable, '-m', 'torch.distributed.run', f'--nproc-per-node={args.gpus}',
                           'tools/train.py', config, '--launcher', 'pytorch', '--work-dir', str(output)]
            if args.resume:
                command += ['--resume']
            run_command(command + extra + shared, args.dry_run)
        if args.mode in ('test', 'train-test'):
            checkpoint = best_checkpoint(output, args.dataset, args.dry_run)
            command = [sys.executable, 'tools/test.py', config, str(checkpoint),
                       '--work-dir', str(output / 'eval'), '--split', split]
            run_command(command + (extra if args.mode == 'test' else []) + shared, args.dry_run)
            if args.dataset == 'wwtp' and args.overwrite and name == 'rpgv_v7' and not args.dry_run:
                marker = output / '.retrained_30k_done'
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text('done\n')


if __name__ == '__main__':
    main()
