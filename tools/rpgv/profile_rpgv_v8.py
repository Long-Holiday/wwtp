#!/usr/bin/env python3
"""Synthetic v8 resource/gradient diagnostics, not an accuracy or causality test."""
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
from tools.rpgv.profile_rpgv_v6 import synchronize

# Named children whose completion marks a stage of the single v8 forward pass.
STAGES = ('rgb_encoder', 'gaussian_predictor', 'gaussian_graph',
          'structure_fusion', 'local_fusion', 'decoder', 'refiner')
MODEL_TYPES = ('RPGVNetV8', 'RPGVRemoteSensingV8')


def synthetic_batch(model, size, batch_size):
    """Three-channel BGR only: v8 rejects any pseudo-geometry channel."""
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
    return model.data_preprocessor(dict(inputs=[item['inputs'] for item in items],
                                        data_samples=[item['data_samples'] for item in items]),
                                   training=True)


def apply_overrides(cfg, args):
    grid = tuple(int(value) for value in args.grid_sizes.split(',') if value.strip())
    cfg.model.gaussian_cfg.grid_sizes = grid
    cfg.model.gaussian_cfg.max_size = args.max_size
    cfg.model.graph_cfg.use_contact = bool(getattr(args, 'use_contact', True))
    # v8.1-only switches; the flag list decides which of them exist.
    for flag, key, owner in (('content_aware', 'content_aware', 'gaussian_cfg'),
                             ('use_edge_value', 'use_edge_value', 'graph_cfg'),
                             ('local_detail', 'local_detail', None)):
        if hasattr(args, flag):
            target = cfg.model if owner is None else cfg.model[owner]
            target[key] = bool(getattr(args, flag))


def node_count(cfg, raster):
    grids = cfg.model.gaussian_cfg.grid_sizes
    return sum(min(value, raster[0]) * min(value, raster[1]) for value in grids)


def stage_timeline(model, device):
    """First-completion time and peak allocation per staged module."""
    timeline, start = {}, time.perf_counter()

    def record(name):
        def hook(module, args, output):
            if name in timeline:
                return
            timeline[name] = dict(
                seconds=time.perf_counter() - start,
                peak_allocated_mib=(torch.cuda.max_memory_allocated(device) / 2**20
                                    if device.type == 'cuda' else None))
        return hook

    targets, handles = [], []
    for name in STAGES:
        if not hasattr(model, name):
            continue
        module = getattr(model, name)
        # ModuleList forwards nothing itself; its members are called directly.
        members = list(module.named_children()) if isinstance(module, torch.nn.ModuleList) else []
        targets.extend([(f'{name}[{index}]', child) for index, child in members] or [(name, module)])
    for name, module in targets:
        handles.append(module.register_forward_hook(record(name)))
    return timeline, handles


def build_model(cfg, device, training):
    model = MODELS.build(cfg.model).to(device).train(training)
    return model


def run_steps(model, batch, args, device, timeline=None):
    training = args.mode == 'train'
    if training:
        optimizer_cfg = copy.deepcopy(args.optim_wrapper)
        optimizer_cfg.accumulative_counts = 1
        if device.type != 'cuda':
            # AMP-only keys are rejected by the plain CPU wrapper; the configured
            # optimizer, learning rate and clipping are unchanged.
            optimizer_cfg.type = 'OptimWrapper'
            for key in ('loss_scale', 'dtype'):
                optimizer_cfg.pop(key, None)
        optimizer = build_optim_wrapper(model, optimizer_cfg).optimizer
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    rows, times = [], []
    for step in range(args.warmup + args.steps):
        if training:
            optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        begin = time.perf_counter()
        with torch.set_grad_enabled(training):
            output = model.loss(**batch) if training else model._run_network(batch['inputs'])
        if training:
            loss = sum(value for key, value in output.items() if 'loss' in key)
            loss.backward()
            missing = [name for name, p in model.named_parameters()
                       if p.requires_grad and p.grad is None]
            if missing:
                raise RuntimeError(f'Parameters without gradients: {missing}')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), error_if_nonfinite=True,
                                                 **args.clip_grad)
            optimizer.step()
        elif not torch.isfinite(output['seg_logits']).all():
            raise RuntimeError('non-finite inference output')
        synchronize(device)
        duration = time.perf_counter() - begin
        row = dict(step=step, seconds=duration, warmup=step < args.warmup)
        if training:
            row.update(metrics={name: float(value.detach()) for name, value in output.items()},
                       unclipped_gradient_l2=float(norm))
        rows.append(row)
        if step >= args.warmup:
            times.append(duration)
        del output
    return rows, times


def gradient_report(model, predictor):
    """Per-parameter gradient L2, so the reach scalar is visible by name."""
    report = {}
    groups = [predictor] + [getattr(model, name) for name in ('gaussian_graph', 'local_fusion')
                            if hasattr(model, name)]
    for group in groups:
        for name, parameter in group.named_parameters():
            report[name] = (None if parameter.grad is None
                            else float(parameter.grad.float().square().sum().sqrt()))
    report['__module_l2'] = {
        name: float(torch.stack([p.grad.float().square().sum() for p in module.parameters()
                                 if p.grad is not None]).sum().sqrt())
        for name, module in model.named_children()
        if any(True for _ in module.parameters())}
    return report


def switch_report(model):
    """Effective v8.1 switches, empty for a model that has none of them."""
    report = {}
    predictor = getattr(model, 'gaussian_predictor', None)
    if hasattr(predictor, 'content_aware'):
        report['content_aware'] = bool(predictor.content_aware)
    graph = getattr(model, 'gaussian_graph', None)
    layers = list(graph.layers) if hasattr(graph, 'layers') else []
    if layers and hasattr(layers[0], 'use_edge_value'):
        report['use_edge_value'] = bool(layers[0].use_edge_value)
    if isinstance(getattr(model, 'local_detail', None), bool):
        report['local_detail'] = model.local_detail
    return report


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
    if cfg.model.type not in MODEL_TYPES:
        raise ValueError('expected a v8 config')
    disable_pretraining(cfg.model)
    cfg.model.data_preprocessor.size = (args.size, args.size)
    apply_overrides(cfg, args)
    training = args.mode == 'train'
    args.optim_wrapper = cfg.optim_wrapper
    args.clip_grad = cfg.optim_wrapper.clip_grad
    model = build_model(cfg, device, training)
    batch = synthetic_batch(model, args.size, args.batch_size)
    timeline, handles = ({}, [])
    if training and args.stages:
        timeline, handles = stage_timeline(model, device)
    rows, times = run_steps(model, batch, args, device)
    for handle in handles:
        handle.remove()
    result = dict(status='ok', config=args.config, device=str(device), mode=args.mode,
                  input_shape=list(batch['inputs'].shape), precision='fp32',
                  gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  grid_sizes=list(cfg.model.gaussian_cfg.grid_sizes),
                  max_size=cfg.model.gaussian_cfg.max_size,
                  use_contact=bool(cfg.model.graph_cfg.use_contact),
                  parameters=sum(p.numel() for p in model.parameters()),
                  mean_seconds=statistics.mean(times), steps=rows,
                  scope='Synthetic random initialization; no data I/O, pretrained download or validation. '
                        'Full optimizer update per step, no accumulation/LR schedule. '
                        'Uniform configured optimizer/clipping. Not accuracy or convergence evidence.')
    for key, value in switch_report(model).items():
        result[key] = value
    if training:
        result['gradient_parameter_l2'] = gradient_report(model, model.gaussian_predictor)
    if timeline:
        result['stage_timeline'] = timeline
    if device.type == 'cuda':
        result.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
    return result


def matrix(args):
    """One run per (grid_sizes, contact) cell, plus the node count of each."""
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    register_all_modules(init_default_scope=True)
    MMLogger.get_current_instance().setLevel(logging.ERROR)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    cfg = Config.fromfile(args.config)
    if cfg.model.type not in MODEL_TYPES:
        raise ValueError('expected a v8 config')
    disable_pretraining(cfg.model)
    cfg.model.data_preprocessor.size = (args.size, args.size)
    args.optim_wrapper = cfg.optim_wrapper
    args.clip_grad = cfg.optim_wrapper.clip_grad
    raster = (min(args.size // 4, args.max_size),) * 2
    cells = []
    for grid in ((8,), (4, 8, 16), (20,)):
        for contact in (True, False):
            cell = copy.deepcopy(cfg)
            cell.model.gaussian_cfg.grid_sizes = grid
            cell.model.gaussian_cfg.max_size = args.max_size
            cell.model.graph_cfg.use_contact = contact
            torch.manual_seed(42)
            model = build_model(cell, device, True)
            batch = synthetic_batch(model, args.size, args.batch_size)
            rows, times = run_steps(model, batch, args, device)
            entry = dict(grid_sizes=list(grid), use_contact=contact,
                         nodes=node_count(cell, raster),
                         parameters=sum(p.numel() for p in model.parameters()),
                         mean_seconds=statistics.mean(times))
            if device.type == 'cuda':
                entry.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                             peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
            cells.append(entry)
            del model, batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return dict(status='ok', config=args.config, device=str(device), input_size=args.size,
                batch_size=args.batch_size, max_size=args.max_size, cells=cells,
                scope='Synthetic random initialization; single-process sequential cells, so '
                      'later cells inherit allocator state. Not accuracy evidence.')


def build_parser(default_config='configs/rpgv/v8/rpgv_v8.py'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=default_config)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--mode', choices=('train', 'infer', 'matrix'), default='train')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--grid-sizes', default='4,8,16')
    parser.add_argument('--max-size', type=int, default=64)
    parser.add_argument('--no-contact', dest='use_contact', action='store_false')
    parser.add_argument('--stages', action='store_true',
                        help='record per-stage completion time and peak allocation')
    parser.add_argument('--output', type=Path)
    return parser


def main(argv=None, default_config='configs/rpgv/v8/rpgv_v8.py', extra_arguments=None):
    parser = build_parser(default_config)
    if extra_arguments is not None:
        extra_arguments(parser)
    args = parser.parse_args(argv)
    if args.size < 32 or min(args.batch_size, args.steps) < 1 or args.warmup < 0:
        parser.error('size >= 32, positive batch/steps and nonnegative warmup required')
    if args.max_size < 1:
        parser.error('max_size must be positive')
    if args.output and args.output.exists():
        parser.error('output already exists; choose a new diagnostic path')
    try:
        result = matrix(args) if args.mode == 'matrix' else profile(args)
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
