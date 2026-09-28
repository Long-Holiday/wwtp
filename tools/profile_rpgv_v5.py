#!/usr/bin/env python3
"""Bounded real-data CUDA AMP update check; never saves trained checkpoints."""
import argparse
import json
import time
from pathlib import Path

import torch
from mmengine.config import Config
from mmseg.registry import DATASETS, MODELS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/v5/rpgv_v5.py')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 4:
        parser.error('positive batch size and at least 4 steps required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; run inside the existing GPU Docker image')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    cfg.model.rgb_encoder.init_cfg = None
    model = MODELS.build(cfg.model).cuda().train()
    # All comparison models use identical NEW 1m samples and preprocessing.
    data_cfg = Config.fromfile('configs/v5/rpgv_v5.py')
    dataset = DATASETS.build(data_cfg.train_dataloader.dataset)
    positive = next(i for i in range(len(dataset))
                    if not Path(dataset.get_data_info(i)['img_path']).name.startswith('neg_'))
    indices = [positive] if args.batch_size == 1 else [positive, 0]
    indices = (indices * args.batch_size)[:args.batch_size]
    samples = [dataset[i] for i in indices]
    batch = model.data_preprocessor(dict(
        inputs=[x['inputs'] for x in samples],
        data_samples=[x['data_samples'] for x in samples]), training=True)
    assert batch['inputs'].shape == (args.batch_size, 5, 1024, 1024)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=128.0)
    first = next(model.rgb_encoder.parameters())
    before = first.detach().clone()
    durations, rows = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.float16):
            losses = model.loss(batch['inputs'], batch['data_samples'])
            loss, _ = model.parse_losses(losses)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not torch.isfinite(norm):
            raise RuntimeError(f'Nonfinite gradients at step {step}')
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        if step >= 2:
            durations.append(duration)
        rows.append(dict(step=step, loss=float(loss.detach()), grad_norm=float(norm),
                         seconds=duration, scale=scaler.get_scale()))
    difference = float((first.detach() - before).abs().max())
    if difference == 0:
        raise RuntimeError('RGB backbone did not update')
    result = dict(
        config=args.config, batch_size=args.batch_size, input_size=[1024, 1024],
        dataset=data_cfg.data_root, device=torch.cuda.get_device_name(),
        torch_version=torch.__version__, parameters=sum(p.numel() for p in model.parameters()),
        mean_step_seconds=sum(durations) / len(durations),
        images_per_second=args.batch_size * len(durations) / sum(durations),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        backbone_max_update=difference, steps=rows,
        scope='Random initialization, fixed real batch, AMP forward/loss/backward/AdamW; '
              'two warmup steps excluded. No data I/O, validation, accumulation or convergence claim.')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'steps'}, indent=2))


if __name__ == '__main__':
    main()
