#!/usr/bin/env python3
"""Audit saved WWTP masks; no inference, threshold tuning or postprocessing.

Small-component counts are diagnostics, not a recommendation to delete objects.
Use validation masks for model development and keep test data held out.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def topology(mask, valid, small_area):
    labels, count = ndi.label(mask & valid, np.ones((3, 3)))
    areas = np.bincount(labels.ravel())[1:]
    # Exclude exterior/ignored background from hole counts.
    background, _ = ndi.label(~mask & valid, np.ones((3, 3)))
    exterior = np.unique(np.concatenate([
        background[0], background[-1], background[:, 0], background[:, -1],
        background[ndi.binary_dilation(~valid)],
    ]))
    hole_areas = np.bincount(background.ravel())
    hole_areas[exterior] = 0
    hole_areas[0] = 0
    return dict(components=int(count), small_components=int((areas < small_area).sum()),
                holes=int((hole_areas > 0).sum()), hole_pixels=int(hole_areas.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery', type=Path, default=Path('work_dirs/wwtp_inference_gallery'))
    parser.add_argument('--data-root', type=Path, default=Path('wwtp_semantic_dataset'))
    parser.add_argument('--models', nargs='+', default=['rpgv-net', 'deeplabv3plus'])
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--small-area', type=int, default=256)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.small_area < 1:
        parser.error('--small-area must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.gallery / 'manifest.json').read_text())
    report = dict(split=args.split, small_area_px=args.small_area, connectivity=8,
                  selection=manifest['selection'], models={})
    for model in args.models:
        rows = []
        for path in sorted((args.gallery / 'masks' / args.split / model).glob('*.png')):
            gt = np.asarray(Image.open(args.data_root / 'annotations' / args.split / path.name))
            pred = np.asarray(Image.open(path)) > 0
            valid, fg = gt != 255, gt == 1
            if pred.shape != gt.shape:
                raise ValueError(f'shape mismatch: {path}')
            pred &= valid
            labels, _ = ndi.label(pred, np.ones((3, 3)))
            areas = np.bincount(labels.ravel())
            overlap = np.bincount(labels[fg], minlength=len(areas))
            detached = (overlap == 0)
            detached[0] = False
            rows.append(dict(image=path.name, gt=topology(fg, valid, args.small_area),
                             pred=topology(pred, valid, args.small_area),
                             tp=int((pred & fg).sum()), fp=int((pred & ~fg).sum()),
                             fn=int((~pred & fg).sum()),
                             detached_fp_components=int(detached.sum()),
                             detached_fp_pixels=int(areas[detached].sum()),
                             negative=not bool(fg.any())))
        if not rows or len(rows) != manifest['splits'][args.split]:
            raise ValueError(f'incomplete masks for {model}: {len(rows)}')
        totals = {k: sum(r[k] for r in rows) for k in
                  ('tp', 'fp', 'fn', 'detached_fp_components', 'detached_fp_pixels')}
        totals['iou'] = totals['tp'] / max(totals['tp'] + totals['fp'] + totals['fn'], 1)
        totals['negative_images'] = sum(r['negative'] for r in rows)
        totals['negative_images_with_fp'] = sum(r['negative'] and r['fp'] > 0 for r in rows)
        totals['negative_fp_pixels'] = sum(r['fp'] for r in rows if r['negative'])
        for side in ('gt', 'pred'):
            totals[side] = {k: sum(r[side][k] for r in rows) for k in rows[0][side]}
        report['models'][model] = dict(
            checkpoint=next(m['checkpoint'] for m in manifest['models'] if m['slug'] == model),
            images=len(rows), totals=totals, per_image=rows)
        print(model, json.dumps(totals), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
