#!/usr/bin/env python3
"""Restart RPGV Stage 3 from the 40k best weights for 30k iterations.

This is intentionally isolated from the original staged-training workflow:
the checkpoint is loaded as model weights only, a fresh optimizer and fresh
parameter schedulers are constructed, and all outputs go to a new work dir.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / 'configs/experiments/rpgv_stage3_joint.py'
DEFAULT_CHECKPOINT = (
    REPO_ROOT / 'work_dirs/rpgv_staged/stage3_joint'
    / 'best_binary_Foreground_IoU_iter_40000.pth')
DEFAULT_WORK_DIR = REPO_ROOT / 'work_dirs/rpgv_stage3_restart_30k'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint',
        default=str(DEFAULT_CHECKPOINT),
        help='Stage-3 checkpoint whose model weights will be loaded.')
    parser.add_argument(
        '--work-dir',
        default=str(DEFAULT_WORK_DIR),
        help='New, empty directory for continuation logs and checkpoints.')
    parser.add_argument(
        '--base-lr', type=float, default=1.5e-4,
        help='Base learning rate for the fresh AdamW optimizer.')
    parser.add_argument(
        '--max-iters', type=int, default=30000,
        help='Number of new micro-batch iterations.')
    parser.add_argument(
        '--val-interval', type=int, default=2000,
        help='Validation and checkpoint interval.')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate paths and print the continuation settings only.')
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint}')
    if args.base_lr <= 0.0:
        raise ValueError('--base-lr must be positive')
    if args.max_iters <= 0:
        raise ValueError('--max-iters must be positive')
    if args.val_interval <= 0:
        raise ValueError('--val-interval must be positive')
    if work_dir == checkpoint.parent:
        raise ValueError(
            'The continuation work dir must differ from the source '
            f'checkpoint directory: {checkpoint.parent}')
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(
            f'Refusing to mix outputs into non-empty directory: {work_dir}. '
            'Choose a new --work-dir.')
    return checkpoint, work_dir


def build_config(
    args: argparse.Namespace,
    checkpoint: Path,
    work_dir: Path,
) -> Config:
    cfg = Config.fromfile(str(DEFAULT_CONFIG))

    # load_from restores model parameters only. In particular, it does not
    # restore the old optimizer, scheduler state, iteration, or message hub.
    cfg.load_from = str(checkpoint)
    cfg.resume = False
    cfg.pop('required_previous_stage', None)
    cfg.work_dir = str(work_dir)
    cfg.launcher = args.launcher

    # All parameters have already seen Stage-3 training. Keep the newly
    # integrated validation/fusion blocks at the base LR while protecting the
    # pretrained RGB and geometry paths with discriminative multipliers.
    cfg.optim_wrapper = dict(
        type='AmpOptimWrapper',
        loss_scale='dynamic',
        accumulative_counts=8,
        optimizer=dict(
            type='AdamW',
            lr=args.base_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01),
        clip_grad=dict(max_norm=10.0, norm_type=2),
        paramwise_cfg=dict(
            custom_keys=dict(
                rgb_encoder=dict(lr_mult=0.05),
                global_film=dict(lr_mult=0.25),
                rgb_aux_head=dict(lr_mult=0.25),
                rgb_boundary_head=dict(lr_mult=0.25),
                rectifier=dict(lr_mult=0.25),
                geometry_encoder=dict(lr_mult=0.25),
                geometry_pyramid=dict(lr_mult=0.25),
                geometry_aux_head=dict(lr_mult=0.25),
                decoder=dict(lr_mult=0.25),
                refiner=dict(lr_mult=0.25),
                detail_refiner=dict(lr_mult=0.25),
                norm=dict(decay_mult=0.0)),
            norm_decay_mult=0.0))

    # Warm-restart gently, then reduce LR only when validation IoU has really
    # plateaued. Unlike the old PolyLR schedule, this never forces LR to zero.
    cfg.param_scheduler = [
        dict(
            type='LinearLR',
            start_factor=0.1,
            by_epoch=False,
            begin=0,
            end=500),
        dict(
            type='ReduceOnPlateauLR',
            monitor='binary/Foreground_IoU',
            rule='greater',
            factor=0.5,
            patience=3,
            threshold=0.1,
            threshold_rule='abs',
            cooldown=1,
            min_value=1e-6,
            by_epoch=True),
    ]

    cfg.train_cfg = dict(
        type='IterBasedTrainLoop',
        max_iters=args.max_iters,
        val_interval=args.val_interval)
    cfg.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=args.val_interval,
        max_keep_ckpts=3,
        save_best='binary/Foreground_IoU',
        rule='greater')

    # Do not inherit or inject early stopping: this isolated experiment must
    # complete exactly the requested 30k new iterations. Best-checkpoint
    # selection still protects against a worse final iteration.
    cfg.custom_hooks = []
    return cfg


def print_summary(
    args: argparse.Namespace,
    checkpoint: Path,
    work_dir: Path,
) -> None:
    full_updates, remainder = divmod(args.max_iters, 8)
    print('RPGV Stage-3 isolated continuation')
    print(f'  source checkpoint : {checkpoint}')
    print(f'  output directory  : {work_dir}')
    print(f'  fresh AdamW LR     : {args.base_lr:g}')
    print(f'  new iterations     : {args.max_iters}')
    update_summary = str(full_updates)
    if remainder:
        update_summary += f' (+ one partial update from {remainder} micro-batches)'
    print(f'  optimizer updates  : {update_summary}')
    print(f'  validation interval: {args.val_interval}')
    print('  early stopping     : disabled (always run all new iterations)')
    print('  resume state       : model weights only (fresh optimizer/schedulers)')


def main() -> None:
    args = parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    checkpoint, work_dir = validate_args(args)
    cfg = build_config(args, checkpoint, work_dir)
    print_summary(args, checkpoint, work_dir)
    if args.dry_run:
        print('Dry run complete; no files were written.')
        return

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
