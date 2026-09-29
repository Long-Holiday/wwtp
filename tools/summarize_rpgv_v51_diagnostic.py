#!/usr/bin/env python3
"""Summarize same-checkpoint val changes with paired image bootstrap intervals."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='docs/analysis/rpgv_v51_val_diagnostic.json')
    parser.add_argument('--output', default='docs/analysis/rpgv_v51_val_diagnostic_summary.json')
    args = parser.parse_args()
    result = json.loads(Path(args.input).read_text())
    names = [row['image'] for row in result['per_image']['full']]
    generator = np.random.default_rng(42)
    indices = generator.integers(0, len(names), (2000, len(names)))
    bootstraps, summary = {}, {}
    for name, rows in result['per_image'].items():
        if names != [row['image'] for row in rows]:
            raise ValueError('paired image order differs')
        counts = np.array([[row['tp'], row['fp'], row['fn']] for row in rows], dtype=np.float64)
        totals = counts[indices].sum(axis=1)
        bootstraps[name] = 100 * totals[:, 0] / totals.sum(axis=1).clip(1)
        metrics = result['metrics'][name]
        summary[name] = dict(
            iou=metrics['binary']['Foreground_IoU'],
            bf1=metrics['binary']['Boundary_F1'],
            hd95_m=metrics['binary']['HD95_m'],
            small_components=metrics['topology']['pred_small_components'],
            holes=metrics['topology']['pred_holes'],
            detached_fp_components=metrics['topology']['detached_fp_components'],
            detached_fp_pixels=metrics['topology']['detached_fp_pixels'])
    for name, row in summary.items():
        row['full_minus_variant_iou_pp'] = summary['full']['iou'] - row['iou']
        row['paired_bootstrap_95pct_pp'] = np.percentile(
            bootstraps['full'] - bootstraps[name], [2.5, 97.5]).tolist()
    report = dict(
        evaluated_images=len(names), bootstrap_replicates=2000, seed=42,
        interpretation='Paired image resampling, same trained checkpoint and validation split; '
                       'intervals describe sample sensitivity, not training-seed variance '
                       'or independent out-of-sample confirmation.',
        modes=summary)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
