#!/usr/bin/env python3
"""Throughput and peak-memory sweep for RPGV v8 cells on a single GPU.

Synthetic random data and random initialisation, so every number here is a
hardware-utilisation measurement, not an accuracy or convergence claim:
forward/backward cost does not depend on the content of the batch. A cell is a
(micro batch, precision, gradient checkpointing) combination and reports seconds
per step, seconds per image and peak CUDA memory. Each training cell also checks
that the loss and every parameter gradient stayed finite, which is the cheap
guard against a mixed-precision cell that silently trains nothing.
"""
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

import wwtpseg  # noqa: F401
from wwtpseg.engine.runtime import disable_pretraining

# 'tf32' is fp32 with TF32 matmuls enabled; the others are autocast dtypes.
PRECISION_DTYPES = {'fp32': None, 'tf32': None, 'bf16': torch.bfloat16, 'fp16': torch.float16}
DEFAULT_TRAIN_CELLS = ('2:fp32:cp,4:fp32:cp,8:fp32:cp,8:fp32:nocp,8:tf32:cp,'
                       '2:bf16:cp,4:bf16:cp,8:bf16:cp,8:bf16:nocp,12:bf16:nocp,'
                       '8:fp16:cp,8:fp16:nocp')
DEFAULT_INFER_CELLS = '1:fp32,4:fp32,8:fp32,4:bf16,8:bf16'


def parse_train_cell(text):
    fields = text.split(':')
    if len(fields) != 3 or fields[1] not in PRECISION_DTYPES or fields[2] not in ('cp', 'nocp'):
        raise argparse.ArgumentTypeError(
            f'expected batch:precision:cp|nocp (precision one of {tuple(PRECISION_DTYPES)}), got {text!r}')
    return dict(batch_size=int(fields[0]), precision=fields[1], with_cp=fields[2] == 'cp')


def parse_infer_cell(text):
    fields = text.split(':')
    if len(fields) != 2 or fields[1] not in PRECISION_DTYPES:
        raise argparse.ArgumentTypeError(
            f'expected batch:precision (precision one of {tuple(PRECISION_DTYPES)}), got {text!r}')
    return dict(batch_size=int(fields[0]), precision=fields[1])


def packed_batch(size, batch_size):
    """BGR-only synthetic batch, mirroring the geometry of the v8 profiler."""
    packer = TRANSFORMS.build(dict(type='PackSegInputs'))
    y, x = np.mgrid[:size, :size].astype(np.float32) / max(size - 1, 1)
    rng = np.random.default_rng(42)
    items = []
    for index in range(batch_size):
        circle = (x - 0.4 - 0.04 * (index % 2))**2 + (y - 0.5)**2 < 0.12**2
        ditch = (x > 0.6) & (x < 0.64) & (y > 0.15) & (y < 0.85)
        rectangle = (x > 0.7) & (x < 0.9) & (y > 0.2) & (y < 0.65)
        image = rng.uniform(0, 255, (size, size, 3)).astype(np.float32)
        items.append(packer(dict(
            img=image, gt_seg_map=(circle | ditch | rectangle).astype(np.uint8),
            seg_fields=['gt_seg_map'], img_shape=(size, size), ori_shape=(size, size),
            scale_factor=(1.0, 1.0), reduce_zero_label=False)))
    return dict(inputs=[item['inputs'] for item in items],
                data_samples=[item['data_samples'] for item in items])


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def set_precision(precision, device):
    """TF32 is a global backend flag; it is left on only for tf32 cells."""
    if device.type != 'cuda':
        return
    enabled = precision == 'tf32'
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def build_train_cell(cfg, cell, device):
    cell_cfg = copy.deepcopy(cfg)
    cell_cfg.model.rgb_encoder.with_cp = cell['with_cp']
    torch.manual_seed(42)
    model = MODELS.build(cell_cfg.model).to(device).train()
    wrapper_cfg = copy.deepcopy(cell_cfg.optim_wrapper)
    wrapper_cfg.accumulative_counts = 1
    if cell['precision'] in ('bf16', 'fp16'):
        wrapper_cfg.type = 'AmpOptimWrapper'
        wrapper_cfg.dtype = 'bfloat16' if cell['precision'] == 'bf16' else 'float16'
        if cell['precision'] == 'bf16':
            # tools/train.py does the same: bf16 needs no loss scaling.
            wrapper_cfg.loss_scale = 1.0
    return model, build_optim_wrapper(model, wrapper_cfg)


def run_train_cell(cfg, cell, data, args, device):
    model, optim_wrapper = build_train_cell(cfg, cell, device)
    peak = dict(peak_allocated_mib=None, peak_reserved_mib=None)
    times, loss_value, finite = [], None, True
    for step in range(1, args.warmup + args.steps + 1):
        optim_wrapper.zero_grad()
        synchronize(device)
        begin = time.perf_counter()
        with optim_wrapper.optim_context(model):
            losses = model.loss(**data)
            loss = sum(value for key, value in losses.items() if 'loss' in key)
        optim_wrapper.update_params(loss)
        synchronize(device)
        duration = time.perf_counter() - begin
        loss_value = float(loss.detach())
        if step == args.warmup and device.type == 'cuda':
            # Steady-state peak: after the allocator has taken the optimizer states.
            torch.cuda.reset_peak_memory_stats(device)
        if step > args.warmup:
            times.append(duration)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (parameter.grad is None
                                        or not torch.isfinite(parameter.grad).all()):
            finite = False
            break
    if device.type == 'cuda':
        peak = dict(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
    record = dict(cell=f"{cell['batch_size']}:{cell['precision']}:{'cp' if cell['with_cp'] else 'nocp'}",
                  mode='train', batch_size=cell['batch_size'], precision=cell['precision'],
                  with_cp=cell['with_cp'], steps=len(times),
                  mean_seconds=statistics.mean(times),
                  median_seconds=statistics.median(times),
                  seconds_per_image=statistics.mean(times) / cell['batch_size'],
                  seconds_all_steps=times, last_loss=loss_value, finite=finite, **peak)
    del model, optim_wrapper
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return record


def run_infer_cell(cfg, cell, packed, args, device):
    cell_cfg = copy.deepcopy(cfg)
    cell_cfg.model.rgb_encoder.with_cp = False
    torch.manual_seed(42)
    model = MODELS.build(cell_cfg.model).to(device).eval()
    dtype = PRECISION_DTYPES[cell['precision']]
    autocast = torch.autocast('cuda', dtype=dtype or torch.bfloat16,
                              enabled=device.type == 'cuda' and dtype is not None)
    peak = dict(peak_allocated_mib=None, peak_reserved_mib=None)
    times, finite = [], True
    with torch.no_grad(), autocast:
        for step in range(1, args.warmup + args.steps + 1):
            synchronize(device)
            begin = time.perf_counter()
            outputs = model.val_step(packed)
            synchronize(device)
            duration = time.perf_counter() - begin
            if step == args.warmup and device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            if step > args.warmup:
                times.append(duration)
            finite = finite and all(
                bool(np.isfinite(sample.pred_sem_seg.data).all()) for sample in outputs)
        if device.type == 'cuda':
            peak = dict(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
    record = dict(cell=f"{cell['batch_size']}:{cell['precision']}", mode='infer',
                  batch_size=cell['batch_size'], precision=cell['precision'], with_cp=False,
                  steps=len(times), mean_seconds=statistics.mean(times),
                  median_seconds=statistics.median(times),
                  seconds_per_image=statistics.mean(times) / cell['batch_size'],
                  seconds_all_steps=times, finite=finite, **peak)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/rpgv/v8/rpgv_v8.py')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--train-cells', type=lambda text: [parse_train_cell(f) for f in text.split(',')],
                        default=[parse_train_cell(f) for f in DEFAULT_TRAIN_CELLS.split(',')])
    parser.add_argument('--infer-cells', type=lambda text: [parse_infer_cell(f) for f in text.split(',')],
                        default=[parse_infer_cell(f) for f in DEFAULT_INFER_CELLS.split(',')])
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-infer', action='store_true')
    parser.add_argument('--json', type=Path, help='also write the records to this path')
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    register_all_modules(init_default_scope=True)
    MMLogger.get_current_instance().setLevel(logging.ERROR)
    cfg = Config.fromfile(args.config)
    disable_pretraining(cfg.model)
    cfg.model.data_preprocessor.size = (args.size, args.size)

    records, order = [], []
    if not args.skip_train:
        order.append('train')
        for cell in args.train_cells:
            set_precision(cell['precision'], device)
            data = None
            try:
                model_probe, _ = build_train_cell(cfg, cell, device)
                data = model_probe.data_preprocessor(
                    packed_batch(args.size, cell['batch_size']), training=True)
                del model_probe
                torch.cuda.empty_cache()
                record = run_train_cell(cfg, cell, data, args, device)
                del data
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                record = dict(cell=f"{cell['batch_size']}:{cell['precision']}:"
                                   f"{'cp' if cell['with_cp'] else 'nocp'}", mode='train',
                              batch_size=cell['batch_size'], precision=cell['precision'],
                              with_cp=cell['with_cp'], oom=True)
            records.append(record)
            print(json.dumps(record), flush=True)
    if not args.skip_infer:
        order.append('infer')
        for cell in args.infer_cells:
            set_precision(cell['precision'], device)
            try:
                record = run_infer_cell(cfg, cell, packed_batch(args.size, cell['batch_size']),
                                        args, device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                record = dict(cell=f"{cell['batch_size']}:{cell['precision']}", mode='infer',
                              batch_size=cell['batch_size'], precision=cell['precision'], oom=True)
            records.append(record)
            print(json.dumps(record), flush=True)
    result = dict(status='ok', config=args.config, device=str(device), size=args.size,
                  precision_order=['fp32', 'tf32 (matmul+cudnn TF32)', 'bf16 autocast', 'fp16 autocast'],
                  gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  parameters=None, records=records,
                  scope='Synthetic random data and initialisation; fwd/bwd cost only, no data I/O, '
                        'no accumulation, no LR schedule, no validation. Hardware utilisation only.')
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
