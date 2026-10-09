#!/usr/bin/env python3
"""Bounded synthetic v6 profiling; no pretrained download or trained checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.optim import build_optim_wrapper
from mmseg.registry import MODELS, TRANSFORMS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401,E402
from wwtpseg.engine.runtime import disable_pretraining


def synthetic_batch(model, size, batch_size, cache_targets):
    packer = TRANSFORMS.build(dict(
        type='PackRPGVContourInputs', contour_truncation=model.contour_truncation)
        if cache_targets else dict(type='PackSegInputs'))
    y, x = np.mgrid[:size, :size].astype(np.float32) / max(size - 1, 1)
    rng = np.random.default_rng(42)
    items = []
    for index in range(batch_size):
        circle = (x - 0.4 - 0.04 * (index % 2))**2 + (y - 0.5)**2 < 0.12**2
        rectangle = (x > 0.6) & (x < 0.8) & (y > 0.2) & (y < 0.65)
        image = rng.uniform(0, 255, (size, size, 5)).astype(np.float32)
        image[..., 3] = (.2 + .15 * x + .1 * y + .3 * circle + .15 * rectangle) * 255
        image[..., 4] = 0.85 * 255
        items.append(packer(dict(
            img=image, gt_seg_map=(circle | rectangle).astype(np.uint8),
            seg_fields=['gt_seg_map'], img_shape=(size, size), ori_shape=(size, size),
            scale_factor=(1.0, 1.0), reduce_zero_label=False)))
    return model.data_preprocessor(dict(inputs=[item['inputs'] for item in items],
                                        data_samples=[item['data_samples'] for item in items]),
                                   training=True)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def profile(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use a working GPU container or --device cpu')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    register_all_modules(init_default_scope=True)
    MMLogger.get_current_instance().setLevel(logging.ERROR)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
    config = Config.fromfile(args.config)
    if config.model.type != 'RPGVNetV6':
        raise ValueError('this profiler is for RPGVNetV6 configs')
    disable_pretraining(config.model)
    config.model.data_preprocessor.size = (args.size, args.size)
    if args.with_cp != 'config':
        config.model.rgb_encoder.with_cp = args.with_cp == 'on'
    if args.attention != 'config':
        config.model.rgb_encoder.type = ('MixVisionTransformerSDPA' if args.attention == 'sdpa'
                                         else 'MixVisionTransformer')
    model = MODELS.build(config.model).to(device)
    model.train(args.mode == 'train')
    batch = synthetic_batch(model, args.size, args.batch_size,
                            args.mode == 'train' and not args.no_cache_targets)
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.precision]
    amp = args.precision != 'fp32'
    result = dict(
        config=str(args.config), mode=args.mode, input_shape=list(batch['inputs'].shape),
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        torch_version=torch.__version__, precision=args.precision,
        parameters=sum(p.numel() for p in model.parameters()),
        backbone=config.model.rgb_encoder.type, with_cp=config.model.rgb_encoder.get('with_cp', False),
        cached_sdf=args.mode == 'train' and not args.no_cache_targets,
        warmup_steps=args.warmup, measured_steps=args.steps,
        scope='Random initialization, synthetic inputs; no data I/O or validation. '
              'Training measures a full optimizer update without gradient accumulation; '
              'configured parameter groups and clipping are retained. Not an accuracy/convergence test.')
    if args.mode == 'train':
        wrapper_config = copy.deepcopy(config.optim_wrapper)
        wrapper_config.type = 'OptimWrapper'
        wrapper_config.pop('loss_scale', None)
        wrapper_config.accumulative_counts = 1
        optimizer = build_optim_wrapper(model, wrapper_config).optimizer
        initial_scale = config.optim_wrapper.get('loss_scale', {}).get('init_scale', 128.0)
        scaler = torch.cuda.amp.GradScaler(
            enabled=device.type == 'cuda' and args.precision == 'fp16', init_scale=initial_scale)
        first = next(model.rgb_encoder.parameters())
        before = first.detach().clone()
        clip = config.optim_wrapper.get('clip_grad', {})
    else:
        scaler = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    optimizer_enabled = args.mode == 'train'
    times, rows = [], []
    for step in range(args.warmup + args.steps):
        if optimizer_enabled:
            optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        start = time.perf_counter()
        with torch.set_grad_enabled(optimizer_enabled), torch.autocast(
                device_type=device.type, dtype=dtype if amp else torch.bfloat16, enabled=amp):
            if optimizer_enabled:
                outputs = model.loss(**batch)
                loss = sum(value for name, value in outputs.items() if 'loss' in name)
            else:
                outputs = model._run_network(batch['inputs'])
        if optimizer_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
            if missing:
                raise RuntimeError(f'Parameters missing gradients: {missing}')
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip.get('max_norm', float('inf')),
                norm_type=clip.get('norm_type', 2), error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
        elif not torch.isfinite(outputs['seg_logits']).all():
            raise RuntimeError('Non-finite inference output')
        synchronize(device)
        duration = time.perf_counter() - start
        if step >= args.warmup:
            times.append(duration)
        row = dict(step=step, warmup=step < args.warmup, seconds=duration)
        if optimizer_enabled:
            row.update(losses={name: float(value.detach()) for name, value in outputs.items()},
                       unclipped_grad_norm=float(norm), amp_scale=scaler.get_scale())
        rows.append(row)
        del outputs
    result.update(status='ok', mean_seconds=statistics.mean(times),
                  median_seconds=statistics.median(times),
                  images_per_second=args.batch_size / statistics.mean(times), steps=rows)
    if device.type == 'cuda':
        result.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
    if args.mode == 'train':
        result['backbone_max_update'] = float((first.detach() - before).abs().max())
        if result['backbone_max_update'] == 0:
            raise RuntimeError('RGB backbone did not update')
        result['gradient_group_l2'] = {}
        for name, module in model.named_children():
            grads = [p.grad.detach().float().square().sum()
                     for p in module.parameters() if p.grad is not None]
            result['gradient_group_l2'][name] = float(torch.stack(grads).sum().sqrt()) if grads else 0
        if model.use_geometry:
            result['branch_gains'] = [stage.fusion.branch_gains.detach().cpu().tolist()
                                      for stage in model.geometry_stages]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rpgv/v6/rpgv_v6.py')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--mode', choices=('train', 'infer'), default='train')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp16')
    parser.add_argument('--steps', type=int, default=3, help='Measured steps, excluding warmup')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--with-cp', choices=('config', 'on', 'off'), default='config')
    parser.add_argument('--attention', choices=('config', 'sdpa', 'weights'), default='config')
    parser.add_argument('--no-cache-targets', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.size < 32 or args.batch_size < 1 or args.steps < 1 or args.warmup < 0:
        parser.error('size >= 32, positive batch/steps and nonnegative warmup required')
    if args.device == 'cpu' and args.precision == 'fp16':
        parser.error('CPU profiling supports fp32/bf16; use --precision fp32')
    try:
        result = profile(args)
        status = 0
    except Exception as error:
        result = dict(status='oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'error',
                      config=args.config, mode=args.mode, message=str(error))
        if torch.cuda.is_available() and args.device == 'cuda':
            result.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        status = 1
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end='', flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
