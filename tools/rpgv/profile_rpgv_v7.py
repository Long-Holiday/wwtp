#!/usr/bin/env python3
"""Synthetic v7 resource/gradient diagnostics, not an accuracy or causality test."""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.optim import build_optim_wrapper
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401
from wwtpseg.engine.runtime import disable_pretraining
from tools.rpgv.profile_rpgv_v6 import synthetic_batch, synchronize


def profile(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    register_all_modules(init_default_scope=True)
    MMLogger.get_current_instance().setLevel(logging.ERROR)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    cfg = Config.fromfile(args.config)
    if cfg.model.type not in ('RPGVNetV7', 'RPGVRemoteSensingV7'):
        raise ValueError('expected a v7 config')
    disable_pretraining(cfg.model)
    cfg.model.data_preprocessor.size = (args.size, args.size)
    model = MODELS.build(cfg.model).to(device).train(args.mode == 'train')
    batch = synthetic_batch(model, args.size, args.batch_size,
                            cache_targets=args.mode == 'train' and model.refiner.num_classes == 1)
    training = args.mode == 'train'
    if training:
        optimizer_cfg = copy.deepcopy(cfg.optim_wrapper)
        optimizer_cfg.accumulative_counts = 1
        optimizer = build_optim_wrapper(model, optimizer_cfg).optimizer
        tracked = {name: next(module.parameters()) for name, module in model.named_children()
                   if any(True for _ in module.parameters())}
        before = {name: value.detach().clone() for name, value in tracked.items()}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    rows, times = [], []
    for step in range(args.warmup + args.steps):
        if training:
            optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            output = model.loss(**batch) if training else model._run_network(batch['inputs'])
        if training:
            loss = sum(v for key, v in output.items() if 'loss' in key)
            loss.backward()
            missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
            if missing:
                raise RuntimeError(f'Parameters without gradients: {missing}')
            gradients = {}
            for name, module in model.named_children():
                values = [p.grad.float().square().sum() for p in module.parameters() if p.grad is not None]
                gradients[name] = float(torch.stack(values).sum().sqrt()) if values else 0.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), error_if_nonfinite=True,
                                                 **cfg.optim_wrapper.clip_grad)
            optimizer.step()
        elif not torch.isfinite(output['seg_logits']).all():
            raise RuntimeError('non-finite inference output')
        synchronize(device)
        duration = time.perf_counter() - start
        row = dict(step=step, seconds=duration, warmup=step < args.warmup)
        if training:
            row.update(metrics={name: float(value.detach()) for name, value in output.items()},
                       unclipped_gradient_l2=float(norm), gradient_group_l2=gradients)
        rows.append(row)
        if step >= args.warmup:
            times.append(duration)
        del output
    result = dict(status='ok', config=args.config, device=str(device), mode=args.mode,
                  input_shape=list(batch['inputs'].shape), precision='fp32',
                  gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  parameters=sum(p.numel() for p in model.parameters()),
                  mean_seconds=statistics.mean(times), steps=rows,
                  scope='Synthetic random initialization; no data I/O, pretrained download or validation. '
                        'Full optimizer update per step, no accumulation/LR schedule. '
                        'Uniform configured optimizer/clipping. Not accuracy or convergence evidence.')
    if training:
        result['module_first_parameter_max_update'] = {
            name: float((value.detach() - before[name]).abs().max()) for name, value in tracked.items()}
        if result['module_first_parameter_max_update']['rgb_encoder'] == 0:
            raise RuntimeError('RGB backbone did not update')
    if device.type == 'cuda':
        result.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rpgv/v7/rpgv_v7.py')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--mode', choices=('train', 'infer'), default='train')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.size < 32 or min(args.batch_size, args.steps) < 1 or args.warmup < 0:
        parser.error('size >= 32, positive batch/steps and nonnegative warmup required')
    if args.output and args.output.exists():
        parser.error('output already exists; choose a new diagnostic path')
    try:
        result = profile(args)
    except Exception as error:
        result = dict(status='oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'error',
                      config=args.config, input_size=args.size, batch_size=args.batch_size,
                      mode=args.mode, message=str(error))
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='', flush=True)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
