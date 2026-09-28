#!/usr/bin/env python3
"""Fast, deliberately biased screening funnel for RPGV-v2 ablations.

The workflow is intentionally separate from ``run_rpgv_v2_ablations.py``:

1. build one fixed, foreground-coverage-stratified mini validation set;
2. evaluate every switch using the same trained Full checkpoint;
3. warm-start the best switches and fine-tune at a smaller crop size;
4. write the commands for a clean, full-budget confirmation run.

Quick-screen measurements are useful for deciding what to train next.  They
must not be mixed with the independent, full-budget ablation table.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.evaluator import BaseMetric
from mmengine.runner import Runner
from mmseg.registry import DATASETS, METRICS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401  Register the local dataset, model and transforms.


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / 'configs' / 'ablations_v2'
FULL = 'full'
PAPER = (
    'no_rgr',
    'no_frequency_validation',
    'unweighted_fusion',
    'no_contour',
    'no_structure_regularization',
)
PROGRESSIVE = (
    'progressive_base',
    'progressive_rgr',
    'progressive_dfgv',
    'progressive_weighted',
    'progressive_contour',
)
GROUPS = {
    'paper': PAPER,
    'core': PAPER,
    'progressive': PROGRESSIVE,
    'all': tuple(dict.fromkeys((*PROGRESSIVE, *PAPER))),
}
METRIC = 'binary/Foreground_IoU'


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value) and value.numel() == 1:
        return value.item()
    raise TypeError(f'{type(value).__name__} is not JSON serializable')


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + '\n',
        encoding='utf-8')
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def expand_variants(items: Sequence[str]) -> list[str]:
    names: list[str] = []
    for item in items:
        names.extend(GROUPS.get(item, (item,)))
    names = list(dict.fromkeys(names))
    missing = [name for name in names if not (CONFIG_ROOT / f'{name}.py').is_file()]
    if missing:
        raise ValueError(f'unknown ablation variants: {missing}')
    return names


def resolve_checkpoint(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        candidates = sorted(
            path.glob('best_binary_Foreground_IoU_iter_*.pth'),
            key=lambda item: item.stat().st_mtime)
        if not candidates:
            pointer = path / 'last_checkpoint'
            if pointer.is_file():
                target = Path(pointer.read_text(encoding='utf-8').strip())
                path = target if target.is_absolute() else ROOT / target
            else:
                raise FileNotFoundError(f'no best checkpoint or last_checkpoint in {path}')
        else:
            path = candidates[-1]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _extract_mask(sample, name: str) -> torch.Tensor:
    value = sample[name] if isinstance(sample, dict) else getattr(sample, name)
    if isinstance(value, dict):
        value = value['data']
    elif hasattr(value, 'data'):
        value = value.data
    return torch.as_tensor(value).squeeze().detach().cpu()


@METRICS.register_module(force=True)
class QuickBinaryOverlapMetric(BaseMetric):
    """Cheap foreground overlap metric with no EDT/Hausdorff computation."""

    default_prefix = 'binary'

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        del data_batch
        for sample in data_samples:
            prediction = _extract_mask(sample, 'pred_sem_seg')
            target = _extract_mask(sample, 'gt_sem_seg')
            valid = target != 255
            prediction = (prediction == 1) & valid
            target = (target == 1) & valid
            self.results.append(dict(
                tp=int((prediction & target).sum()),
                fp=int((prediction & ~target & valid).sum()),
                fn=int((~prediction & target).sum()),
                tn=int((~prediction & ~target & valid).sum()),
            ))

    def compute_metrics(self, results: list[dict]) -> dict[str, float]:
        totals = {key: sum(row[key] for row in results) for key in ('tp', 'fp', 'fn', 'tn')}
        tp, fp, fn, tn = (totals[key] for key in ('tp', 'fp', 'fn', 'tn'))

        def divide(numerator, denominator):
            return float(numerator / denominator) if denominator else 0.0

        iou = divide(tp, tp + fp + fn)
        dice = divide(2 * tp, 2 * tp + fp + fn)
        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        accuracy = divide(tp + tn, tp + fp + fn + tn)
        return {
            'Foreground_IoU': 100.0 * iou,
            'Dice': 100.0 * dice,
            'F1': 100.0 * dice,
            'Precision': 100.0 * precision,
            'Recall': 100.0 * recall,
            'Pixel_Accuracy': 100.0 * accuracy,
        }


def config_path(variant: str) -> Path:
    path = CONFIG_ROOT / f'{variant}.py'
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolved_config_hash(path: Path) -> str:
    return text_hash(Config.fromfile(path).pretty_text)


def validation_records(path: Path) -> list[dict]:
    """Return records in the exact order used by the configured dataset."""
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(path)
    dataset_cfg = copy.deepcopy(cfg.val_dataloader.dataset)
    dataset_cfg.pop('indices', None)
    dataset = DATASETS.build(dataset_cfg)
    rows = []
    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        annotation = Path(info['seg_map_path'])
        image = Path(info['img_path'])
        mask = cv2.imread(str(annotation), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f'could not read validation mask: {annotation}')
        if mask.ndim == 3:
            mask = mask[..., 0]
        valid = mask != 255
        valid_count = int(valid.sum())
        coverage = float(((mask == 1) & valid).sum() / valid_count) if valid_count else 0.0
        rows.append(dict(
            index=index,
            image=str(image.resolve()),
            annotation=str(annotation.resolve()),
            foreground_fraction=coverage,
        ))
    return rows


def stratified_selection(records: Sequence[dict], size: int, seed: int) -> list[dict]:
    """Sample evenly from four foreground-coverage quartiles."""
    if size <= 0:
        raise ValueError('mini validation size must be positive')
    if size > len(records):
        raise ValueError(f'mini validation size {size} exceeds dataset size {len(records)}')
    ordered = sorted(records, key=lambda row: (row['foreground_fraction'], row['image']))
    selected: list[dict] = []
    selected_indices: set[int] = set()
    for quartile in range(4):
        begin = len(ordered) * quartile // 4
        end = len(ordered) * (quartile + 1) // 4
        bucket = ordered[begin:end]
        quota = size // 4 + int(quartile < size % 4)
        ranked = sorted(
            bucket,
            key=lambda row: hashlib.sha256(
                f'{seed}:{row["image"]}'.encode('utf-8')).hexdigest())
        for row in ranked[:quota]:
            copied = dict(row, coverage_quartile=quartile)
            selected.append(copied)
            selected_indices.add(row['index'])
    if len(selected) < size:
        remaining = [row for row in records if row['index'] not in selected_indices]
        remaining.sort(key=lambda row: hashlib.sha256(
            f'{seed}:fill:{row["image"]}'.encode('utf-8')).hexdigest())
        selected.extend(dict(row, coverage_quartile=None) for row in remaining[:size - len(selected)])
    return sorted(selected, key=lambda row: row['index'])


def prepare_manifest(
    output: Path,
    size: int,
    seed: int,
    rebuild: bool = False,
) -> dict:
    path = output / 'mini_val.json'
    records = validation_records(config_path(FULL))
    identity = dict(
        schema=1,
        size=size,
        seed=seed,
        dataset_size=len(records),
        dataset_config_hash=resolved_config_hash(config_path(FULL)),
    )
    if path.is_file() and not rebuild:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        for key, value in identity.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    f'{path} was built for a different dataset/options; '
                    'use --rebuild-mini-val or a fresh --work-root')
        current = {row['index']: row for row in records}
        for selected in manifest['samples']:
            row = current.get(selected['index'])
            if row is None or row['image'] != selected['image']:
                raise RuntimeError(f'dataset order changed since {path} was created')
        return manifest
    samples = stratified_selection(records, size, seed)
    manifest = dict(**identity, samples=samples)
    atomic_json(path, manifest)
    return manifest


def manifest_hash(manifest: dict) -> str:
    return text_hash(json.dumps(manifest, sort_keys=True, ensure_ascii=False))


def mini_indices(manifest: dict) -> list[int]:
    return [int(row['index']) for row in manifest['samples']]


def _set_dataloader_runtime(dataloader, batch_size: int, workers: int) -> None:
    dataloader.batch_size = batch_size
    dataloader.num_workers = workers
    dataloader.persistent_workers = workers > 0


def evaluation_config(
    source_config: Path,
    checkpoint: Path,
    indices: Sequence[int],
    work_dir: Path,
    batch_size: int,
    workers: int,
) -> Config:
    cfg = Config.fromfile(source_config)
    cfg.model.rgb_encoder.init_cfg = None
    cfg.load_from = str(checkpoint)
    cfg.resume = False
    cfg.launcher = 'none'
    cfg.work_dir = str(work_dir)
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    cfg.test_dataloader.dataset.indices = list(indices)
    _set_dataloader_runtime(cfg.test_dataloader, batch_size, workers)
    cfg.test_evaluator = [dict(type='QuickBinaryOverlapMetric')]
    cfg.test_cfg = dict(type='TestLoop')
    return cfg


def evaluate_variant(
    stage: str,
    variant: str,
    config: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    manifest: dict,
    output: Path,
    batch_size: int,
    workers: int,
) -> dict:
    directory = output / stage / variant
    result_path = directory / 'metrics.json'
    identity = dict(
        schema=1,
        stage=stage,
        variant=variant,
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=resolved_config_hash(config),
        mini_val_sha256=manifest_hash(manifest),
        sample_count=manifest['size'],
    )
    if result_path.is_file():
        record = json.loads(result_path.read_text(encoding='utf-8'))
        if record.get('identity') != identity:
            raise RuntimeError(f'evaluation identity changed: {result_path}')
        print(f'[SKIP evaluated] {stage} {variant}', flush=True)
        return record

    print(f'[EVAL] {stage} {variant} <- {checkpoint}', flush=True)
    cfg = evaluation_config(
        config, checkpoint, mini_indices(manifest), directory / 'runner',
        batch_size, workers)
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    record = dict(identity=identity, metrics=metrics)
    atomic_json(result_path, record)
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def metric_value(record: dict, name: str = METRIC) -> float:
    try:
        return float(record['metrics'][name])
    except KeyError as error:
        raise KeyError(f'{name} is missing from {record}') from error


def rank_records(records: Iterable[dict], exclude_full: bool = False) -> list[dict]:
    rows = [record for record in records
            if not exclude_full or record['identity']['variant'] != FULL]
    return sorted(rows, key=lambda row: (-metric_value(row), row['identity']['variant']))


def write_stage_summary(stage: str, records: Sequence[dict], output: Path) -> dict:
    ranked = rank_records(records)
    full = next((record for record in records if record['identity']['variant'] == FULL), None)
    full_score = metric_value(full) if full else None
    rows = []
    for rank, record in enumerate(ranked, 1):
        metrics = record['metrics']
        score = metric_value(record)
        rows.append(dict(
            rank=rank,
            variant=record['identity']['variant'],
            foreground_iou=score,
            dice=float(metrics['binary/Dice']),
            precision=float(metrics['binary/Precision']),
            recall=float(metrics['binary/Recall']),
            delta_from_full=None if full_score is None else score - full_score,
            checkpoint=record['identity']['checkpoint'],
        ))
    summary = dict(stage=stage, metric=METRIC, rows=rows)
    atomic_json(output / f'{stage}_summary.json', summary)
    return summary


def _patch_pipeline(pipeline, crop_size: int, thumbnail_size: int, resize_base: int) -> None:
    found = set()
    for transform in pipeline:
        kind = transform.get('type')
        if kind == 'GenerateGlobalThumbnail':
            transform.size = (thumbnail_size, thumbnail_size)
            found.add(kind)
        elif kind == 'RandomResize':
            transform.scale = (resize_base, resize_base)
            found.add(kind)
        elif kind == 'RandomForegroundCrop':
            transform.crop_size = (crop_size, crop_size)
            found.add(kind)
    required = {'GenerateGlobalThumbnail', 'RandomResize', 'RandomForegroundCrop'}
    if missing := required - found:
        raise RuntimeError(f'training pipeline lacks required transforms: {sorted(missing)}')


def fast_finetune_config(
    variant: str,
    source_checkpoint: Path,
    train_work_dir: Path,
    manifest: dict,
    *,
    seed: int,
    iterations: int,
    crop_size: int,
    thumbnail_size: int,
    resize_base: int,
    train_batch_size: int,
    eval_batch_size: int,
    accumulation: int,
    workers: int,
    learning_rate: float,
    checkpoint_interval: int,
) -> Config:
    if iterations <= 0 or crop_size <= 0 or thumbnail_size <= 0:
        raise ValueError('iterations and image sizes must be positive')
    if min(train_batch_size, eval_batch_size, accumulation) <= 0:
        raise ValueError('batch sizes and accumulation must be positive')
    if learning_rate <= 0:
        raise ValueError('learning rate must be positive')

    cfg = Config.fromfile(config_path(variant))
    cfg.model.rgb_encoder.init_cfg = None
    cfg.model.data_preprocessor.size = (crop_size, crop_size)
    cfg.model.global_thumbnail_size = thumbnail_size
    _patch_pipeline(
        cfg.train_dataloader.dataset.pipeline,
        crop_size=crop_size,
        thumbnail_size=thumbnail_size,
        resize_base=resize_base)
    for transform in cfg.val_dataloader.dataset.pipeline:
        if transform.get('type') == 'GenerateGlobalThumbnail':
            transform.size = (thumbnail_size, thumbnail_size)

    cfg.load_from = str(source_checkpoint)
    cfg.resume = False
    cfg.pop('required_previous_stage', None)
    cfg.work_dir = str(train_work_dir)
    cfg.randomness = dict(seed=seed, deterministic=False)
    cfg.env_cfg.cudnn_benchmark = True
    cfg.enable_early_stopping = False
    cfg.custom_hooks = [
        hook for hook in cfg.get('custom_hooks', [])
        if hook.get('type') not in ('EarlyStoppingHook', 'RPGVEarlyStoppingHook')]

    _set_dataloader_runtime(cfg.train_dataloader, train_batch_size, workers)
    cfg.val_dataloader.dataset.indices = mini_indices(manifest)
    _set_dataloader_runtime(cfg.val_dataloader, eval_batch_size, workers)
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)

    cfg.optim_wrapper.accumulative_counts = accumulation
    cfg.optim_wrapper.optimizer.lr = learning_rate
    cfg.auto_scale_lr.enable = False
    warmup = min(100, max(1, iterations // 10))
    linear = next(item for item in cfg.param_scheduler if item.type == 'LinearLR')
    poly = next(item for item in cfg.param_scheduler if item.type == 'PolyLR')
    linear.begin, linear.end = 0, warmup
    poly.begin, poly.end = warmup, iterations
    cfg.train_cfg.max_iters = iterations
    cfg.train_cfg.val_interval = iterations
    cfg.val_evaluator = [dict(
        type='IoUMetric', iou_metrics=['mIoU', 'mDice'], nan_to_num=0)]
    cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
    cfg.default_hooks.checkpoint.interval = min(checkpoint_interval, iterations)
    cfg.default_hooks.checkpoint.max_keep_ckpts = 2
    cfg.default_hooks.checkpoint.save_best = 'mIoU'
    cfg.default_hooks.checkpoint.rule = 'greater'
    return cfg


def _checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.rsplit('_', 1)[-1])
    except ValueError:
        return -1


def finetune_checkpoint(train_dir: Path, iterations: int) -> Path | None:
    best = sorted(train_dir.glob('best_mIoU_iter_*.pth'), key=_checkpoint_iteration)
    if best:
        return best[-1].resolve()
    final = train_dir / f'iter_{iterations}.pth'
    return final.resolve() if final.is_file() else None


def run_finetune(
    variant: str,
    source_checkpoint: Path,
    source_sha256: str,
    manifest: dict,
    output: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    directory = output / 'finetune' / variant
    train_dir = directory / 'train'
    generated_config = directory / 'config.py'
    metadata_path = directory / 'identity.json'
    identity = dict(
        schema=1,
        variant=variant,
        source_checkpoint=str(source_checkpoint),
        source_checkpoint_sha256=source_sha256,
        source_config_sha256=resolved_config_hash(config_path(variant)),
        mini_val_sha256=manifest_hash(manifest),
        seed=args.seed,
        iterations=args.train_iters,
        crop_size=args.crop_size,
        thumbnail_size=args.thumbnail_size,
        resize_base=args.resize_base,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        accumulation=args.accumulative_counts,
        workers=args.num_workers,
        learning_rate=args.learning_rate,
        checkpoint_interval=args.checkpoint_interval,
    )
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding='utf-8'))
        if previous != identity:
            raise RuntimeError(
                f'fine-tune identity changed in {directory}; use a fresh --work-root')
    else:
        atomic_json(metadata_path, identity)

    cfg = fast_finetune_config(
        variant, source_checkpoint, train_dir, manifest,
        seed=args.seed,
        iterations=args.train_iters,
        crop_size=args.crop_size,
        thumbnail_size=args.thumbnail_size,
        resize_base=args.resize_base,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        accumulation=args.accumulative_counts,
        workers=args.num_workers,
        learning_rate=args.learning_rate,
        checkpoint_interval=args.checkpoint_interval)
    config_text = cfg.pretty_text
    if generated_config.is_file() and generated_config.read_text(encoding='utf-8') != config_text:
        raise RuntimeError(f'generated config changed: {generated_config}')
    if not generated_config.is_file():
        atomic_text(generated_config, config_text)

    checkpoint = finetune_checkpoint(train_dir, args.train_iters)
    if checkpoint is None:
        command = [
            sys.executable, str(ROOT / 'tools' / 'train.py'), str(generated_config),
            '--work-dir', str(train_dir),
        ]
        if train_dir.is_dir() and any(train_dir.iterdir()):
            pointer = train_dir / 'last_checkpoint'
            if not args.resume_finetune or not pointer.is_file():
                raise RuntimeError(
                    f'incomplete fine-tune in {train_dir}; pass --resume-finetune '
                    'when a valid last_checkpoint exists, or use a fresh --work-root')
            command.append('--resume')
        print('[TRAIN]', shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        checkpoint = finetune_checkpoint(train_dir, args.train_iters)
        if checkpoint is None:
            raise RuntimeError(f'fine-tune produced no final checkpoint in {train_dir}')
    else:
        print(f'[SKIP trained] {variant}: {checkpoint}', flush=True)
    return checkpoint, generated_config


def read_summary(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def select_from_summary(summary: dict, count: int) -> list[str]:
    if count <= 0:
        raise ValueError('selection count must be positive')
    candidates = [row for row in summary['rows'] if row['variant'] != FULL]
    return [row['variant'] for row in candidates[:count]]


def markdown_table(title: str, summary: dict) -> list[str]:
    lines = [f'## {title}', '',
             '| Rank | Variant | Foreground IoU | Dice | Delta vs Full |',
             '|---:|---|---:|---:|---:|']
    for row in summary['rows']:
        delta = row['delta_from_full']
        lines.append(
            f'| {row["rank"]} | {row["variant"]} | '
            f'{row["foreground_iou"]:.4f} | {row["dice"]:.4f} | '
            f'{"-" if delta is None else f"{delta:+.4f}"} |')
    lines.append('')
    return lines


def official_command(winners: Sequence[str], args: argparse.Namespace, output: Path) -> list[str]:
    variants = list(winners) or [FULL]
    official_root = args.official_work_root or output / 'official_confirmation'

    def portable(path: Path) -> str:
        path = Path(path)
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    return [
        'python',
        portable(ROOT / 'scripts' / 'run_rpgv_v2_ablations.py'),
        'run',
        '--variants', *variants,
        '--seeds', str(args.seed),
        '--max-iters', str(args.official_max_iters),
        '--work-root', portable(official_root),
    ]


def write_report(
    output: Path,
    zero_summary: dict | None,
    finetune_summary: dict | None,
    winners: Sequence[str],
    command: Sequence[str] | None,
) -> None:
    lines = ['# RPGV-v2 quick-screen report', '',
             '> Screening results reuse weights, a mini validation set, and a reduced crop. '
             'Do not report them as independent full-budget ablations.', '']
    if zero_summary:
        lines.extend(markdown_table('Zero-shot mini-val', zero_summary))
    if finetune_summary:
        lines.extend(markdown_table('512-crop short fine-tune', finetune_summary))
    if winners:
        lines.extend(['## Selected for official confirmation', '',
                      ', '.join(f'`{name}`' for name in winners), ''])
    if command:
        rendered = shlex.join(command)
        lines.extend(['## Official confirmation command', '', '```bash', rendered, '```', ''])
        script = '#!/usr/bin/env bash\nset -euo pipefail\n\n' + rendered + '\n'
        target = output / 'run_official_confirmation.sh'
        atomic_text(target, script)
        target.chmod(0o755)
    atomic_text(output / 'report.md', '\n'.join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'action', choices=['prepare', 'zero-shot', 'finetune', 'all', 'official'],
        help='all runs the complete quick funnel but only writes the official command')
    parser.add_argument('--checkpoint', type=Path,
                        help='trained Full checkpoint or directory containing it')
    parser.add_argument('--variants', nargs='+', default=['progressive'],
                        help='variant names or one of: progressive, paper, core, all')
    parser.add_argument('--candidates', nargs='+',
                        help='manual fine-tune candidates; otherwise zero-shot top-k is used')
    parser.add_argument('--work-root', type=Path,
                        default=Path('work_dirs/rpgv_v2_quick_screen'))
    parser.add_argument('--mini-val-size', type=int, default=64)
    parser.add_argument('--mini-val-seed', type=int, default=20260927)
    parser.add_argument('--rebuild-mini-val', action='store_true')
    parser.add_argument('--top-k', type=int, default=3,
                        help='number of zero-shot ablations to short-fine-tune')
    parser.add_argument('--official-top-k', type=int, choices=(1, 2), default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-iters', type=int, default=3000)
    parser.add_argument('--crop-size', type=int, default=512)
    parser.add_argument('--thumbnail-size', type=int, default=256)
    parser.add_argument('--resize-base', type=int, default=1024)
    parser.add_argument('--train-batch-size', type=int, default=2)
    parser.add_argument('--eval-batch-size', type=int, default=1)
    parser.add_argument('--accumulative-counts', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--checkpoint-interval', type=int, default=1000)
    parser.add_argument('--resume-finetune', action='store_true')
    parser.add_argument('--official-max-iters', type=int, default=100000)
    parser.add_argument('--official-work-root', type=Path)
    parser.add_argument('--run-official', action='store_true',
                        help='actually start the long formal confirmation run')
    args = parser.parse_args()
    if args.action in {'zero-shot', 'finetune', 'all'} and args.checkpoint is None:
        parser.error('--checkpoint is required for this action')
    if args.top_k <= 0 or args.mini_val_size <= 0 or args.train_iters <= 0:
        parser.error('--top-k, --mini-val-size and --train-iters must be positive')
    if args.num_workers < 0:
        parser.error('--num-workers cannot be negative')
    return args


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    output = args.work_root.expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    # ``official`` only consumes persisted rankings.  It deliberately avoids
    # rebuilding/scanning a mini-val manifest, so it also works when the
    # original funnel used non-default mini-val options.
    if args.action == 'official':
        finetune_summary = read_summary(output / 'finetune_summary.json')
        zero_path = output / 'zero_shot_summary.json'
        zero_summary = read_summary(zero_path) if zero_path.is_file() else None
        winners = select_from_summary(finetune_summary, args.official_top_k)
        command = official_command(winners, args, output)
        write_report(output, zero_summary, finetune_summary, winners, command)
        print('[OFFICIAL]', shlex.join(command), flush=True)
        if args.run_official:
            subprocess.run(command, cwd=ROOT, check=True)
        return

    variants = expand_variants(args.variants)
    manifest = prepare_manifest(
        output, args.mini_val_size, args.mini_val_seed, args.rebuild_mini_val)
    print(f'[MINI-VAL] {manifest["size"]}/{manifest["dataset_size"]} samples -> '
          f'{output / "mini_val.json"}', flush=True)
    if args.action == 'prepare':
        return

    zero_summary = None
    finetune_summary = None
    checkpoint = None
    checkpoint_sha256 = None
    if args.action in {'zero-shot', 'finetune', 'all'}:
        checkpoint = resolve_checkpoint(args.checkpoint)
        checkpoint_sha256 = file_hash(checkpoint)

    if args.action in {'zero-shot', 'all'}:
        records = []
        for variant in list(dict.fromkeys([*variants, FULL])):
            records.append(evaluate_variant(
                'zero_shot', variant, config_path(variant), checkpoint,
                checkpoint_sha256, manifest, output,
                args.eval_batch_size, args.num_workers))
        zero_summary = write_stage_summary('zero_shot', records, output)
        print(f'[SUMMARY] {output / "zero_shot_summary.json"}', flush=True)
        if args.action == 'zero-shot':
            write_report(output, zero_summary, None, [], None)
            return

    if args.action in {'finetune', 'all'}:
        if args.candidates:
            candidates = expand_variants(args.candidates)
        else:
            if zero_summary is None:
                zero_summary = read_summary(output / 'zero_shot_summary.json')
            candidates = select_from_summary(zero_summary, args.top_k)
        candidates = [name for name in dict.fromkeys(candidates) if name != FULL]
        if not candidates:
            raise RuntimeError('no non-Full candidate was selected for fine-tuning')
        print(f'[SHORTLIST] {", ".join(candidates)}', flush=True)

        records = []
        for variant in list(dict.fromkeys([*candidates, FULL])):
            tuned_checkpoint, generated_config = run_finetune(
                variant, checkpoint, checkpoint_sha256, manifest, output, args)
            records.append(evaluate_variant(
                'finetune_eval', variant, generated_config, tuned_checkpoint,
                file_hash(tuned_checkpoint), manifest, output,
                args.eval_batch_size, args.num_workers))
        finetune_summary = write_stage_summary('finetune', records, output)
        winners = select_from_summary(finetune_summary, args.official_top_k)
        command = official_command(winners, args, output)
        write_report(output, zero_summary, finetune_summary, winners, command)
        print(f'[WINNERS] {", ".join(winners)}', flush=True)
        print(f'[REPORT] {output / "report.md"}', flush=True)
        if args.run_official:
            print('[OFFICIAL]', shlex.join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        return

if __name__ == '__main__':
    main()
