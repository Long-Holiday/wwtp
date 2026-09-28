#!/usr/bin/env python3
"""Generate Depth Anything V2 depth for full-view Potsdam RGB tiles."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from tools.remote_sensing.generate_potsdam_geometry import (
    BatchedDepthPredictor, normalize_and_reliability, save_npz)
from wwtpseg.remote_sensing.potsdam_original import SPLIT_TILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path('data/remote_sensing/prepared/potsdam'))
    parser.add_argument('--size', type=int, default=1536)
    parser.add_argument('--model',
                        default='depth-anything/Depth-Anything-V2-Base-hf')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.size <= 0:
        parser.error('--size must be positive')
    output = args.data_root / f'geometry_depth_anything_{args.size}'
    output.mkdir(parents=True, exist_ok=True)
    image_dir = args.data_root / '2_Ortho_RGB' / '2_Ortho_RGB'
    tiles = sorted(set().union(*SPLIT_TILES.values()))
    pending = [tile for tile in tiles if not (
        output / f'top_potsdam_{tile}_RGB.npz').is_file()]
    if pending:
        predictor = BatchedDepthPredictor(args.model, args.device)
        for index, tile in enumerate(pending, 1):
            stem = f'top_potsdam_{tile}_RGB'
            with Image.open(image_dir / f'{stem}.png') as raw:
                image = raw.convert('RGB').resize(
                    (args.size, args.size), Image.Resampling.LANCZOS)
            original = np.asarray(image)
            horizontal = Image.fromarray(np.ascontiguousarray(original[:, ::-1]))
            vertical = Image.fromarray(np.ascontiguousarray(original[::-1, :]))
            scaled = Image.fromarray(cv2.resize(
                original, None, fx=0.75, fy=0.75,
                interpolation=cv2.INTER_AREA))
            size = (args.size, args.size)
            estimates = [
                predictor.predict_batch([image], size)[0],
                predictor.predict_batch([horizontal], size)[0][:, ::-1].copy(),
                predictor.predict_batch([vertical], size)[0][::-1, :].copy(),
                predictor.predict_batch([scaled], size)[0],
            ]
            depth, reliability = normalize_and_reliability(estimates)
            target = output / f'{stem}.npz'
            save_npz(target, depth, reliability)
            print(f'[{index}/{len(pending)}] {tile}', flush=True)
    metadata = dict(source=args.model, method='Depth Anything V2 four-view consistency',
                    output_shape=[args.size, args.size],
                    input='whole 6000x6000 RGB tile resized to output shape',
                    tiles={split: sorted(values) for split, values in SPLIT_TILES.items()})
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError(f'Existing manifest differs: {manifest}')
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Depth Anything geometry prepared: {output} ({len(tiles)} tiles)')


if __name__ == '__main__':
    main()
