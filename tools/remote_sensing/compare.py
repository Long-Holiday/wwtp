#!/usr/bin/env python3
"""Print local MMSeg metrics beside published results; keep protocol caveats visible."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['loveda', 'potsdam'])
    parser.add_argument('metrics', type=Path, help='metrics.json written by run.py eval')
    args = parser.parse_args()
    refs = json.loads(Path(__file__).with_name('references.json').read_text())
    local = json.loads(args.metrics.read_text())
    reference = refs[args.dataset]
    keys = reference['metric_keys']
    values = {}
    for name, key in keys.items():
        if key not in local:
            raise KeyError(f'Missing {key} in {args.metrics}; available: {list(local)}')
        values[name] = float(local[key])
    print(f"Source: {refs['paper']}\n{refs['source']}")
    print(f"Protocol: {reference['protocol']}")
    print('| Model | mIoU (%) | mF1 (%) | local minus paper (mIoU / mF1 pp) |')
    print('|---|---:|---:|---:|')
    print(f"| Local run | {values['mIoU']:.2f} | {values['mF1']:.2f} | — |")
    for row in reference['rows']:
        delta_iou = values['mIoU'] - row['mIoU']
        delta_f1 = values['mF1'] - row['mF1']
        print(f"| {row['model']} | {row['mIoU']:.2f} | {row['mF1']:.2f} | "
              f'{delta_iou:+.2f} / {delta_f1:+.2f} |')
    print('\nThe cited models use their own training, preprocessing, and inference settings. '
          'These are literature reference values, not controlled reimplementations.')


if __name__ == '__main__':
    main()
