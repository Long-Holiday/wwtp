#!/usr/bin/env python3
"""Profile a single crop plus global thumbnail; run only when the GPU is free."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_rpgv_v2_ablations import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    if args.size < 64 or args.size % 32 or args.warmup < 1 or args.iterations < 2:
        parser.error('size must be a multiple of 32 >=64; warmup >=1; iterations >=2')
    if not torch.cuda.is_available():
        parser.error('CUDA is required for this latency protocol')
    register_all_modules(init_default_scope=True)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(args.config)
    if cfg.model.training_stage != 'joint':
        parser.error('profile the joint-stage configuration')
    cfg.model.rgb_encoder.init_cfg = None
    model = MODELS.build(cfg.model)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['state_dict'], strict=True)
    model = model.cuda().eval()
    inputs = torch.rand(1, 5, args.size, args.size, device='cuda') * 255
    metas = [dict(img_shape=(args.size, args.size))]
    elapsed = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for _ in range(args.warmup):
            model.encode_decode(inputs, metas)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.iterations):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model.encode_decode(inputs, metas)
            stop.record()
            stop.synchronize()
            elapsed.append(start.elapsed_time(stop))
    result = dict(protocol='one crop + freshly encoded global thumbnail; FP16; batch=1',
                  excludes='data loading, pseudo-depth generation, topology metrics, whole-scene stitching',
                  gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, size=args.size,
                  global_thumbnail_size=model.global_thumbnail_size,
                  config_sha256=file_hash(args.config),
                  checkpoint_sha256=file_hash(args.checkpoint) if args.checkpoint else None,
                  iterations=args.iterations, milliseconds_mean=statistics.mean(elapsed),
                  milliseconds_median=statistics.median(elapsed),
                  milliseconds_std=statistics.stdev(elapsed),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  parameters=sum(p.numel() for p in model.parameters()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
