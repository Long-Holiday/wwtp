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
import torch
from PIL import Image
from mmengine.config import Config
from wwtpseg.engine.runtime import disable_pretraining
from wwtpseg.experiments import experiment_config


def check_checkpoint(path):
    if not path.is_file():
        raise AssertionError(f'Production training did not save {path.name}')
    state = torch.load(path, map_location='cpu')['state_dict']
    invalid = [name for name, value in state.items() if not torch.isfinite(value).all()]
    if invalid:
        raise AssertionError(f'Checkpoint has non-finite tensors: {invalid}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--config', default=str(experiment_config('wwtp', 'hrnet')),
                        help='Experiment config to exercise without pretrained downloads')
    args = parser.parse_args()
    process_env = dict(os.environ)
    if args.device == 'cpu':
        process_env['CUDA_VISIBLE_DEVICES'] = ''
    with tempfile.TemporaryDirectory(prefix='wwtp-runner-') as directory:
        root = Path(directory)
        cfg = Config.fromfile(args.config)
        needs_geometry = any(step['type'] == 'LoadPseudoGeometry'
                             for step in cfg.train_dataloader.dataset.pipeline)
        disable_pretraining(cfg.model)
        cfg.model.data_preprocessor.size = (64, 64)
        cfg.optim_wrapper.type = 'OptimWrapper'
        cfg.optim_wrapper.pop('loss_scale', None)
        cfg.optim_wrapper.accumulative_counts = 1
        cfg.optim_wrapper.clip_grad = dict(max_norm=10.0, norm_type=2, error_if_nonfinite=True)
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
            pseudo_dir = root / 'data/pseudo_geometry' / split
            if needs_geometry:
                pseudo_dir.mkdir(parents=True)
            for i in range(2):
                rng = np.random.default_rng(i)
                Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(image_dir / f'{i}.png')
                Image.fromarray(rng.integers(0, 2, (64, 64), dtype=np.uint8)).save(mask_dir / f'{i}.png')
                if needs_geometry:
                    np.savez_compressed(pseudo_dir / f'{i}.npz',
                                        depth=rng.random((64, 64), dtype=np.float32),
                                        reliability=np.full((64, 64), 0.85, dtype=np.float32))
            loader = cfg[f'{split}_dataloader']
            loader.num_workers = 0
            loader.persistent_workers = False
            loader.dataset.data_root = str(root / 'data')
            pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations')]
            if needs_geometry:
                pipeline.append(dict(type='LoadPseudoGeometry',
                                     pseudo_root=str(root / 'data/pseudo_geometry'), required=True))
            packer = dict(type='PackSegInputs')
            if cfg.model.type in ('RPGVNetV6', 'RPGVNetV7') and split == 'train':
                packer = next(dict(step) for step in loader.dataset.pipeline
                              if step['type'] == 'PackRPGVContourInputs')
            loader.dataset.pipeline = pipeline + [packer]
        path = root / 'config.py'
        cfg.dump(path)
        subprocess.run([sys.executable, 'tools/train.py', str(path), '--no-tensorboard'], check=True, env=process_env)
        checkpoint = root / 'run/iter_2.pth'
        check_checkpoint(checkpoint)
        subprocess.run([sys.executable, 'tools/test.py', str(path), str(checkpoint)], check=True, env=process_env)
        metrics = json.loads((root / 'run/eval_test/metrics.json').read_text())
        if 'mIoU' not in metrics:
            raise AssertionError('Production test did not write expected metrics')
        cfg.train_cfg.max_iters = 4
        cfg.dump(path)
        subprocess.run([sys.executable, 'tools/train.py', str(path), '--resume', str(checkpoint),
                        '--no-tensorboard'], check=True, env=process_env)
        final_checkpoint = root / 'run/iter_4.pth'
        check_checkpoint(final_checkpoint)
        first = torch.load(checkpoint, map_location='cpu')
        resumed = torch.load(final_checkpoint, map_location='cpu')
        if first['meta']['iter'] != 2 or resumed['meta']['iter'] != 4:
            raise AssertionError('Production resume did not preserve iteration counters')
        if not first.get('optimizer', {}).get('state') or not resumed.get('optimizer', {}).get('state'):
            raise AssertionError('Checkpoints must include initialized optimizer state')
        if cfg.model.type == 'RPGVNetV8':
            key = 'gaussian_predictor.center_head.weight'
            if torch.equal(first['state_dict'][key], resumed['state_dict'][key]):
                raise AssertionError('Gaussian centers did not update during resumed training')
            # The learned center reach must round-trip; a single scalar update can
            # legitimately round to the same value, so only presence is asserted.
            reach = 'gaussian_predictor.center_log_scale'
            if reach not in first['state_dict'] or reach not in resumed['state_dict']:
                raise AssertionError('Gaussian center reach is missing from the checkpoint')
        print('PASS production train/checkpoint/test/resume', flush=True)


if __name__ == '__main__':
    main()
