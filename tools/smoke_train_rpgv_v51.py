#!/usr/bin/env python3
"""Verify a real v5.1 migrated checkpoint through the production AMP trainer.

Runs eight 1024-pixel microsteps with batch 2 and accumulation 4. The custom
hook checks that two AdamW updates reached both inherited RGB and new spatial
weights. It writes checks.json but no training checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'work_dirs/rpgv_v51_initialization/init.pth'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_parameters(runner) -> dict:
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    return {
        'rgb_encoder_first': next(model.rgb_encoder.parameters()),
        'spatial8_output': model.decoder.spatial8.output.weight,
        'spatial4_output': model.decoder.spatial4.output.weight,
    }


def optimizer_step(state: dict) -> int:
    step = state.get('step', 0)
    if torch.is_tensor(step):
        step = step.item()
    return int(step)


@HOOKS.register_module()
class RPGVV51SmokeHook(Hook):
    priority = 'LOWEST'

    def __init__(self, output: str, source: str, source_sha256: str):
        self.output = Path(output)
        self.source = source
        self.source_sha256 = source_sha256
        self.before = None
        self.initial_loss_scale = None

    def before_train(self, runner):
        if runner.iter != 0:
            raise AssertionError(f'migrated smoke must start at iter 0, got {runner.iter}')
        wrapper = runner.optim_wrapper
        if not hasattr(wrapper, 'loss_scaler'):
            raise AssertionError('expected AmpOptimWrapper with a GradScaler')
        self.initial_loss_scale = float(wrapper.loss_scaler.get_scale())
        if not math.isfinite(self.initial_loss_scale):
            raise AssertionError('initial AMP loss scale is non-finite')
        self.before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in tracked_parameters(runner).items()
        }
        for name in ('spatial8_output', 'spatial4_output'):
            if torch.count_nonzero(self.before[name]).item():
                raise AssertionError(f'{name} was not zero in the migrated checkpoint')

    def after_train(self, runner):
        if self.before is None:
            raise AssertionError('before_train did not run')
        if runner.iter != 8:
            raise AssertionError(f'expected 8 real microsteps, got {runner.iter}')
        current = tracked_parameters(runner)
        differences = {}
        for name, parameter in current.items():
            value = parameter.detach().cpu()
            if not torch.isfinite(value).all().item():
                raise AssertionError(f'non-finite trained parameter: {name}')
            differences[name] = float((value - self.before[name]).abs().max().item())
            if differences[name] <= 0:
                raise AssertionError(f'{name} did not update')

        wrapper = runner.optim_wrapper
        optimizer = wrapper.optimizer
        steps = [optimizer_step(state) for state in optimizer.state.values()]
        if not steps or max(steps) != 2:
            raise AssertionError(f'expected two successful AMP updates, steps={steps[:8]}')
        tracked_steps = {
            name: optimizer_step(optimizer.state.get(parameter, {}))
            for name, parameter in current.items()
        }
        if any(step != 2 for step in tracked_steps.values()):
            raise AssertionError(f'tracked weights missed an update: {tracked_steps}')
        final_scale = float(wrapper.loss_scaler.get_scale())
        if not math.isfinite(final_scale) or final_scale <= 0:
            raise AssertionError(f'invalid final AMP loss scale: {final_scale}')
        if not torch.cuda.is_available():
            raise AssertionError('this full-resolution AMP smoke requires CUDA')
        torch.cuda.synchronize()
        checks = dict(
            source_checkpoint=self.source,
            source_checkpoint_sha256=self.source_sha256,
            config=str((self.output.parent / 'smoke_config.py').resolve()),
            microsteps=runner.iter,
            accumulation=4,
            successful_optimizer_updates=max(steps),
            tracked_optimizer_steps=tracked_steps,
            parameter_max_abs_delta=differences,
            amp_loss_scale=dict(initial=self.initial_loss_scale, final=final_scale),
            gpu=torch.cuda.get_device_name(),
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
            input_size=[1024, 1024], batch_size=2,
            train_indices=[0, 400],
            validation=False, checkpoint_saving=False,
        )
        self.output.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n')
        print(f'v5.1 migrated training smoke passed: {self.output}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--work-dir', type=Path,
                        default=Path('work_dirs/rpgv_v51_runner_smoke'))
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    work_dir = args.work_dir.resolve()
    if not torch.cuda.is_available():
        parser.error('CUDA is required; run inside the existing GPU Docker image')
    if not checkpoint.is_file():
        parser.error(f'migrated initialization checkpoint is missing: {checkpoint}')
    if work_dir.exists():
        parser.error(f'work directory already exists: {work_dir}')
    if work_dir == checkpoint.parent or checkpoint.is_relative_to(work_dir):
        parser.error('work directory must be separate from the initialization checkpoint')

    # Check that this is the strict v5-to-v5.1 migration, not arbitrary weights.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    metadata = state.get('meta', {})
    if metadata.get('architecture') != 'RPGVNetV51' or 'state_dict' not in state:
        parser.error('checkpoint lacks v5-to-v5.1 migration metadata and state_dict')
    if metadata.get('optimizer_transferred') is not False:
        parser.error('smoke expects model-only migration without optimizer state')
    del state

    cfg = Config.fromfile(str(ROOT / 'configs/v51/rpgv_v51.py'))
    if not cfg.get('required_previous_stage'):
        parser.error('v5.1 config no longer requires a validated previous stage')
    cfg.model.rgb_encoder.init_cfg = None  # The migrated checkpoint supplies RGB.
    cfg.load_from = str(checkpoint)
    cfg.resume = False
    cfg.work_dir = str(work_dir)
    cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=8, val_interval=1000)
    cfg.train_dataloader.batch_size = 2
    cfg.optim_wrapper.accumulative_counts = 4
    cfg.train_dataloader.dataset.indices = [0, 400]
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.val_dataloader = cfg.val_evaluator = cfg.val_cfg = None
    cfg.test_dataloader = cfg.test_evaluator = cfg.test_cfg = None
    cfg.default_hooks.checkpoint = None
    cfg.default_hooks.logger.interval = 1
    cfg.enable_early_stopping = False
    # Keep the v5.1 AdamW LR, paramwise multipliers and first warmup segment.
    cfg.optim_wrapper.loss_scale = dict(init_scale=128.0)
    cfg.custom_hooks = [dict(
        type='RPGVV51SmokeHook', output=str(work_dir / 'checks.json'),
        source=str(checkpoint), source_sha256=sha256(checkpoint))]

    work_dir.mkdir(parents=True)
    config_path = work_dir / 'smoke_config.py'
    cfg.dump(str(config_path))
    from train import main as train_main
    sys.argv = ['tools/train.py', str(config_path), '--disable-early-stopping']
    train_main()


if __name__ == '__main__':
    main()
