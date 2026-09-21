#!/usr/bin/env python3
"""Summarize the best validation records from RPGV ablation work dirs."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path


METRICS = (
    'binary/Foreground_IoU',
    'binary/Dice',
    'binary/Boundary_F1',
    'binary/HD95_px',
)


def _metric(record: dict, name: str) -> float | None:
    """Read a metric with or without its evaluator prefix."""
    candidates = (name, name.split('/', 1)[-1])
    for candidate in candidates:
        value = record.get(candidate)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _best_record(log_paths: list[Path]) -> dict | None:
    best = None
    best_iou = float('-inf')
    for log_path in log_paths:
        with log_path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f'invalid JSON in {log_path}:{line_number}') from error
                foreground_iou = _metric(record, METRICS[0])
                if foreground_iou is not None and foreground_iou > best_iou:
                    best = record
                    best_iou = foreground_iou
    return best


def collect(root: Path) -> list[dict[str, str]]:
    """Collect one best stage-3 row per direct child experiment."""
    rows = []
    if not root.is_dir():
        raise FileNotFoundError(f'ablation work root does not exist: {root}')
    for experiment_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        stage_dir = experiment_dir / 'stage3_joint'
        search_root = stage_dir if stage_dir.is_dir() else experiment_dir
        record = _best_record(sorted(search_root.rglob('scalars.json')))
        if record is None:
            continue
        row = {
            'experiment': experiment_dir.name,
            'iter': str(record.get('iter', record.get('step', ''))),
        }
        for metric_name in METRICS:
            value = _metric(record, metric_name)
            row[metric_name.split('/', 1)[-1]] = (
                '' if value is None else f'{value:.4f}')
        rows.append(row)
    return rows


def _render_csv(rows: list[dict[str, str]]) -> str:
    columns = ['experiment', 'iter', *(
        metric.split('/', 1)[-1] for metric in METRICS)]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _render_markdown(rows: list[dict[str, str]]) -> str:
    columns = ['experiment', 'iter', *(
        metric.split('/', 1)[-1] for metric in METRICS)]
    header = '| ' + ' | '.join(columns) + ' |'
    separator = '| ' + ' | '.join('---' for _ in columns) + ' |'
    body = [
        '| ' + ' | '.join(row.get(column, '') for column in columns) + ' |'
        for row in rows
    ]
    return '\n'.join([header, separator, *body]) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'root', nargs='?', type=Path,
        default=Path('work_dirs/rpgv_ablations'))
    parser.add_argument('--format', choices=('markdown', 'csv'), default='markdown')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    rows = collect(args.root)
    if not rows:
        raise RuntimeError(f'no validation scalar records found below {args.root}')
    rendered = (
        _render_markdown(rows)
        if args.format == 'markdown' else _render_csv(rows))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
