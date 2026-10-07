#!/usr/bin/env python3
"""Tiny synthetic train -> checkpoint -> test -> resume using production entries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from mmengine.config import Config
from wwtpseg.engine.runtime import disable_pretraining
from wwtpseg.experiments import experiment_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    process_env = dict(os.environ)
    if args.device == 'cpu':
        process_env['CUDA_VISIBLE_DEVICES'] = ''
    with tempfile.TemporaryDirectory(prefix='wwtp-runner-') as directory:
        root = Path(directory)
        cfg = Config.fromfile(experiment_config('wwtp', 'hrnet'))
        disable_pretraining(cfg.model)
        cfg.model.data_preprocessor.size = (64, 64)
        cfg.optim_wrapper.type = 'OptimWrapper'
        cfg.optim_wrapper.pop('loss_scale', None)
        cfg.optim_wrapper.accumulative_counts = 1
        cfg.train_cfg.max_iters = 2
        cfg.train_cfg.val_interval = 2
        cfg.param_scheduler = [dict(type='PolyLR', begin=0, end=4, by_epoch=False, power=0.9)]
        cfg.default_hooks.checkpoint.interval = 2
        cfg.default_hooks.checkpoint.save_best = None
        cfg.default_hooks.logger.interval = 1
        cfg.val_evaluator = cfg.test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
        cfg.env_cfg.dist_cfg.backend = 'gloo' if args.device == 'cpu' else 'nccl'
        cfg.work_dir = str(root / 'run')
        for split in ('train', 'val', 'test'):
            image_dir, mask_dir = root / 'data/images' / split, root / 'data/annotations' / split
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            for i in range(2):
                rng = np.random.default_rng(i)
                Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(image_dir / f'{i}.png')
                Image.fromarray(rng.integers(0, 2, (64, 64), dtype=np.uint8)).save(mask_dir / f'{i}.png')
            loader = cfg[f'{split}_dataloader']
            loader.num_workers = 0
            loader.persistent_workers = False
            loader.dataset.data_root = str(root / 'data')
            loader.dataset.pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'), dict(type='PackSegInputs')]
        path = root / 'config.py'
        cfg.dump(path)
        subprocess.run([sys.executable, 'tools/train.py', str(path)], check=True, env=process_env)
        checkpoint = root / 'run/iter_2.pth'
        if not checkpoint.is_file():
            raise AssertionError('Production training did not save checkpoint')
        subprocess.run([sys.executable, 'tools/test.py', str(path), str(checkpoint)], check=True, env=process_env)
        metrics = json.loads((root / 'run/eval_test/metrics.json').read_text())
        if 'mIoU' not in metrics:
            raise AssertionError('Production test did not write expected metrics')
        cfg.train_cfg.max_iters = 4
        cfg.dump(path)
        subprocess.run([sys.executable, 'tools/train.py', str(path), '--resume', str(checkpoint)], check=True, env=process_env)
        if not (root / 'run/iter_4.pth').is_file():
            raise AssertionError('Resume did not advance to iteration 4')
        print('PASS production train/checkpoint/test/resume', flush=True)


if __name__ == '__main__':
    main()
