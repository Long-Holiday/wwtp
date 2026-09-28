#!/usr/bin/env python3
"""Exercise the real MMEngine entry point with 8 full-resolution microsteps."""
import argparse
import sys
from pathlib import Path

from mmengine.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', default='work_dirs/rpgv_v5_runner_smoke')
    args = parser.parse_args()
    target = Path(args.work_dir)
    if target.exists():
        raise RuntimeError(f'Refusing to overwrite {target}')
    target.mkdir(parents=True)
    cfg = Config.fromfile('configs/v5/rpgv_v5.py')
    cfg.model.rgb_encoder.init_cfg = None
    cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=8, val_interval=1000)
    cfg.optim_wrapper.loss_scale = dict(init_scale=128.0)
    cfg.train_dataloader.dataset.indices = [0, 400]
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.val_dataloader = cfg.val_evaluator = cfg.val_cfg = None
    cfg.test_dataloader = cfg.test_evaluator = cfg.test_cfg = None
    cfg.default_hooks.checkpoint = None
    cfg.default_hooks.logger.interval = 1
    cfg.work_dir = str(target)
    path = target / 'smoke_config.py'
    cfg.dump(str(path))
    # Run the production entry point, including its no-early-stop behavior.
    from train import main as train_main
    sys.argv = ['tools/train.py', str(path), '--disable-early-stopping']
    train_main()


if __name__ == '__main__':
    main()
