"""Rebuild the 1 m v5/SegNeXt training audit from saved logs.

Run in the existing Docker image. Only reads experiment artifacts and writes
the JSON specified by --output; no model inference or training is performed.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
VAL_KEYS = (
    'binary/Foreground_IoU', 'binary/Dice', 'binary/Precision',
    'binary/Recall', 'binary/Boundary_F1', 'binary/Boundary_Precision',
    'binary/Boundary_Recall', 'binary/HD95_px',
    'topology/images', 'topology/negative_images',
    'topology/negative_images_with_fp', 'topology/gt_components',
    'topology/pred_components', 'topology/pred_small_components',
    'topology/pred_holes', 'topology/pred_hole_pixels',
    'topology/detached_fp_components', 'topology/detached_fp_pixels',
    'topology/fp_pixels', 'topology/detached_fp_fraction',
)


def relative(path):
    return str(path.relative_to(ROOT))


def avg(rows, key):
    vals = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return mean(vals) if vals else None


def summarize(name, timestamp):
    directory = ROOT / 'work_dirs' / name
    run = directory / timestamp
    scalar_path = run / 'vis_data/scalars.json'
    config_path = run / 'vis_data/config.py'
    log_path = run / f'{timestamp}.log'
    rows = [json.loads(line) for line in scalar_path.read_text().splitlines()
            if line.strip()]
    validation = [row for row in rows if 'binary/Foreground_IoU' in row]
    training = [row for row in rows if 'loss' in row]
    assert len(validation) == 20 and validation[-1]['step'] == 20000
    best = max(validation, key=lambda row: row['binary/Foreground_IoU'])
    best_checkpoint = directory / f'best_binary_Foreground_IoU_iter_{best["step"]}.pth'
    final_checkpoint = directory / 'iter_20000.pth'
    assert best_checkpoint.is_file() and final_checkpoint.is_file()
    assert log_path.is_file() and config_path.is_file()
    windows = {}
    for label, low, high in (('early', 1, 1000), ('around_best', 11500, 12500),
                             ('late', 19000, 20000)):
        subset = [row for row in training if low <= row['step'] <= high]
        windows[label] = {
            'records': len(subset), 'train_loss_mean': avg(subset, 'loss'),
            'time_per_iter_s_mean': avg(subset, 'time'),
            'peak_memory_mb_mean': avg(subset, 'memory'),
        }
        for key in ('loss_final', 'loss_final_boundary', 'loss_sdf',
                    'decode.loss_ce', 'decode.loss_dice'):
            value = avg(subset, key)
            if value is not None:
                windows[label][key] = value
    def selected(row):
        return {'step': row['step'], **{key: row.get(key) for key in VAL_KEYS}}
    return {
        'name': name, 'config': relative(config_path),
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'log': relative(log_path), 'scalars': relative(scalar_path),
        'best_checkpoint': relative(best_checkpoint),
        'best_checkpoint_size_bytes': best_checkpoint.stat().st_size,
        'final_checkpoint': relative(final_checkpoint),
        'final_checkpoint_size_bytes': final_checkpoint.stat().st_size,
        'train_records': len(training), 'val_records': len(validation),
        'best': selected(best), 'final': selected(validation[-1]),
        'validation_curve': [selected(row) for row in validation],
        'train_windows': windows,
        'val_iou_10k_12k_mean': avg([row for row in validation
                                      if 10000 <= row['step'] <= 12000],
                                     'binary/Foreground_IoU'),
        'val_iou_13k_20k_mean': avg([row for row in validation
                                      if 13000 <= row['step'] <= 20000],
                                     'binary/Foreground_IoU'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='docs/analysis/rpgv_v51_training_audit.json')
    args = parser.parse_args()
    v5 = summarize('rpgv_v5_1m', '20260928_004513')
    segnext = summarize('segnext_1m', '20260928_042114')
    result = {
        'schema': 1, 'split': 'val',
        'metric_units': 'percent except distances in 1 m pixels',
        'runs': {'rpgv_v5_1m': v5, 'segnext_1m': segnext},
        'best_iou_delta_v5_minus_segnext_pp':
            v5['best']['binary/Foreground_IoU'] - segnext['best']['binary/Foreground_IoU'],
        'test_evaluations': [relative(path) for name in ('rpgv_v5_1m', 'segnext_1m')
                             for path in sorted((ROOT / 'work_dirs' / name /
                                                 'test_eval').glob('**/*.json'))],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(f'Wrote {output}: 2 complete training curves, 20 val records each')


if __name__ == '__main__':
    main()
