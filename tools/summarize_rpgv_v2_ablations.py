#!/usr/bin/env python3
"""Report completed ablations with a common early-stop rule with paired-seed deltas."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_rpgv_v2_ablations import read_validation, METRIC, PAPER, PROGRESSIVE
from wwtpseg.engine.early_stopping import plateau_reached

FIELDS = {
    'iou': 'binary/Foreground_IoU', 'dice': 'binary/Dice',
    'precision': 'binary/Precision', 'recall': 'binary/Recall',
    'boundary_f1': 'binary/Boundary_F1', 'hd95_px': 'binary/HD95_px',
    'islands': 'topology/detached_fp_components',
    'island_pixels': 'topology/detached_fp_pixels',
    'holes': 'topology/pred_holes', 'hole_pixels': 'topology/pred_hole_pixels',
    'parameters': 'complexity/parameters',
}


def collect(root, split):
    rows = []
    for path in sorted(Path(root).glob('*/seed_*/*/experiment.json')):
        spec = json.loads(path.read_text())
        directory = path.parent
        final_run = 'joint' if spec.get('schema', 1) >= 2 else 'stage3_joint'
        stage3 = directory / final_run
        validations = read_validation(stage3)
        last = max((r['selected_iter'] for r in validations), default=None)
        complete = True
        for stage_name, budget in spec['budgets'].items():
            marker = directory / stage_name / 'completed.json'
            if not marker.exists():
                complete = False
                break
            completed = json.loads(marker.read_text())
            actual = max((r['selected_iter'] for r in read_validation(directory / stage_name)),
                         default=0)
            if (completed['budget'] != budget or actual == 0 or actual > budget
                    or actual != completed.get('last_val_iter', budget)):
                complete = False
                break
            if spec.get('schema', 1) >= 3 and actual < budget:
                if (not completed.get('stopped_early') or not spec.get('early_stopping')
                        or not plateau_reached(read_validation(directory / stage_name),
                                               spec['early_stopping'])):
                    complete = False
                    break
        row = dict(protocol=spec['protocol'], variant=spec['variant'], seed=spec['seed'],
                   loss_weighting=spec.get('loss_weighting', 'learned' if spec.get('schema') == 2 else 'fixed'),
                   stopped_early=None,
                   status='complete' if complete else 'incomplete' if validations else 'planned',
                   budget=spec['budgets'][final_run], last_val_iter=last, best_iter=None,
                   val_iou=None, val_boundary_f1=None, split=split, delta_val_iou=None, delta_previous_val_iou=None,
                   **{k: None for k in FIELDS})
        if complete:
            done = json.loads((stage3 / 'completed.json').read_text())
            best = done['validation']
            row.update(stopped_early=done.get('stopped_early', False), best_iter=best['selected_iter'], val_iou=best[METRIC],
                       val_boundary_f1=best.get('binary/Boundary_F1'))
            result = directory / f'evaluation_{split}' / 'metrics.json'
            if result.exists():
                evaluation = json.loads(result.read_text())
                if (evaluation['checkpoint_sha256'] != done['checkpoint_sha256']
                        or evaluation['split'] != split):
                    raise RuntimeError(f'stale or mismatched evaluation: {result}')
                row.update({key: evaluation['metrics'].get(metric) for key, metric in FIELDS.items()})
                row['status'] = 'evaluated'
        rows.append(row)
    references = {(r['protocol'], r['seed'], r['budget']): r for r in rows
                  if r['variant'] == 'full' and r['val_iou'] is not None}
    for row in rows:
        reference = references.get((row['protocol'], row['seed'], row['budget']))
        if reference and row['val_iou'] is not None:
            row['delta_val_iou'] = row['val_iou'] - reference['val_iou']
    paired = {(r['protocol'], r['seed'], r['budget'], r['variant']): r for r in rows}
    for row in rows:
        if row['variant'] in PROGRESSIVE[1:] and row['val_iou'] is not None:
            previous = PROGRESSIVE[PROGRESSIVE.index(row['variant']) - 1]
            reference = paired.get((row['protocol'], row['seed'], row['budget'], previous))
            if reference and reference['val_iou'] is not None:
                row['delta_previous_val_iou'] = row['val_iou'] - reference['val_iou']
    return rows


def aggregate(rows):
    output = []
    for protocol, variant in sorted({(r['protocol'], r['variant']) for r in rows}):
        subset = [r for r in rows if (r['protocol'], r['variant']) == (protocol, variant)]
        result = dict(protocol=protocol, variant=variant, planned_seeds=len(subset))
        for key in ('val_iou', 'delta_val_iou', 'delta_previous_val_iou', *FIELDS):
            values = [r[key] for r in subset if r[key] is not None]
            result[key] = dict(n=len(values), mean=statistics.mean(values) if values else None,
                               std=statistics.stdev(values) if len(values) > 1 else None)
        output.append(result)
    return output


def display(value):
    return '-' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value)


def render_paper(rows, split, variants=PAPER):
    labels = dict(zip(PAPER, (
        'RPGV-v2 (Full)', 'w/o RGR', 'w/o DFGV', 'w/o reliability weighting',
        'w/o contour correction', 'w/o final-mask structural constraints')))
    labels.update(dict(zip(PROGRESSIVE[:-1], (
        'RGB + raw geometry', '+ RGR', '+ DFGV', '+ reliability weighting',
        '+ contour correction'))))
    if variants == PROGRESSIVE:
        labels['full'] = '+ final-mask structural constraints (Full)'
    grouped = aggregate(rows)
    protocol = next((name for name in ('single_stage', 'paper_reuse', 'full', 'joint')
                     if any(g['protocol'] == name for g in grouped)), 'full')
    groups = {g['variant']: g for g in grouped if g['protocol'] == protocol}
    protocol_note = ('Completed Full checkpoint reused; ablations retrain from the first affected '
                     'stage with a common early-stop rule.' if protocol == 'paper_reuse'
                     else 'Independent single-stage joint training; no checkpoint reuse. See run records for loss weighting and actual stopping iterations.'
                     if protocol == 'single_stage' else 'Independent full-stage training.')
    lines = [f'# RPGV-v2 paper ablation — {split}', '',
             f'{protocol_note} Missing evaluations are shown as -.', '',
             '| Model | n | IoU ↑ (%) | Boundary F1 ↑ (%) | HD95 ↓ (px) |',
             '| --- | --- | --- | --- | --- |']
    for variant in variants:
        if variant not in groups:
            continue
        group = groups[variant]
        values = []
        for key in ('iou', 'boundary_f1', 'hd95_px'):
            result = group[key]
            value = display(result['mean'])
            if result['std'] is not None:
                value += ' ± ' + display(result['std'])
            values.append(value)
        lines.append(f"| {labels[variant]} | {group['iou']['n']} | " + ' | '.join(values) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--output-prefix', type=Path)
    args = parser.parse_args()
    rows = collect(args.root, args.split)
    if not rows:
        parser.error('no experiment manifests found')
    prefix = args.output_prefix or args.root / f'summary_{args.split}'
    prefix.parent.mkdir(parents=True, exist_ok=True)
    grouped = aggregate(rows)
    prefix.with_suffix('.json').write_text(json.dumps(dict(runs=rows, groups=grouped), indent=2) + '\n')
    with prefix.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = ('protocol', 'variant', 'seed', 'status', 'loss_weighting', 'stopped_early', 'last_val_iter', 'best_iter',
               'val_iou', 'delta_val_iou', 'delta_previous_val_iou', 'iou', 'boundary_f1', 'hd95_px', 'island_pixels', 'holes')
    lines = [f'# RPGV v2 ablations — evaluation split: {args.split}', '',
             'Missing results are `-`. Only stages with completion records enter means and deltas; '
             'recorded early stops are complete.', '',
             '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines += ['| ' + ' | '.join(display(r[k]) for k in columns) + ' |' for r in rows]
    lines += ['', '## Seed statistics', '',
              'Sample SD (n−1); one seed has no SD. Deltas pair each seed with full in the same protocol.', '',
              '| protocol | variant | n / planned | Val IoU mean ± SD | paired n | ΔVal IoU mean ± SD |',
              '| --- | --- | --- | --- | --- | --- |']
    for group in grouped:
        val, delta = group['val_iou'], group['delta_val_iou']
        lines.append(f"| {group['protocol']} | {group['variant']} | {val['n']} / {group['planned_seeds']} | "
                     f"{display(val['mean'])} ± {display(val['std'])} | {delta['n']} | "
                     f"{display(delta['mean'])} ± {display(delta['std'])} |")
    prefix.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    if any(row['protocol'] in ('single_stage', 'paper_reuse', 'full') for row in rows):
        paper = prefix.with_name(prefix.name + '_paper').with_suffix('.md')
        paper.write_text(render_paper(rows, args.split))
    if any(row['variant'] in PROGRESSIVE[:-1] for row in rows):
        progressive = prefix.with_name(prefix.name + '_progressive').with_suffix('.md')
        progressive.write_text(render_paper(rows, args.split, PROGRESSIVE))
    print(f'Saved {prefix}.{{md,csv,json}}')


if __name__ == '__main__':
    main()
