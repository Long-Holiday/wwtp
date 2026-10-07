#!/usr/bin/env python3
"""Export private-test predictions with dataset-native IDs, without fake metrics."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from mmengine.config import Config, DictAction
from mmseg.apis import init_model, inference_model

from wwtpseg.engine.runtime import prepare_runtime
from tools.remote_sensing.prepare import UAVID_COLORS


def submission_mask(dataset_type, prepared_name, ids):
    """Return submission-relative path and mask in the benchmark's encoding."""
    group, separator, filename = Path(prepared_name).stem.partition('_')
    if not separator or not filename:
        raise ValueError(f'Expected prepared domain/sequence prefix: {prepared_name}')
    if dataset_type == 'LoveDADataset':
        if group not in ('Urban', 'Rural') or np.any(ids > 6):
            raise ValueError('Invalid LoveDA prediction')
        # The official prediction script saves argmax IDs 0..6, even though
        # source ground truth uses 1..7. Submission filenames are flat.
        return Path(f'{filename}.png'), Image.fromarray(ids.astype(np.uint8))
    if dataset_type == 'UAVidDataset':
        if not group.startswith('seq') or np.any(ids > 7):
            raise ValueError('Invalid UAVid prediction')
        return Path(group) / 'Labels' / f'{filename}.png', Image.fromarray(np.asarray(UAVID_COLORS, dtype=np.uint8)[ids])
    raise ValueError(f'Unsupported public dataset: {dataset_type}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    prepare_runtime(cfg, inference=True)
    paths = sorted((Path(cfg.data_root) / 'img_dir' / 'test').glob('*.png'))
    if not paths:
        raise FileNotFoundError(f'No prepared test images under {cfg.data_root}')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    model = init_model(cfg, args.checkpoint, device=args.device)
    # inference_model uses the configured annotation-free pipeline.
    model.cfg.test_pipeline = cfg.predict_pipeline
    dataset_type = cfg.predict_dataset.type
    for path in paths:
        result = inference_model(model, str(path))
        ids = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        relative_path, image = submission_mask(dataset_type, path.name, ids)
        target = args.output / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f'Duplicate submission filename: {target}')
        image.save(target)
    print(f'Exported {len(paths)} masks to {args.output}')


if __name__ == '__main__':
    main()
