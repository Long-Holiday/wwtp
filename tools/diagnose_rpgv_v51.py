#!/usr/bin/env python3
"""Same-checkpoint, full-validation diagnosis of RPGV v5 geometry and contour.

The four modes share one RGB encoder pass per image. This is a diagnostic of a
fixed trained model, not a separately trained ablation. Only val is accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.structures import PixelData
from mmseg.registry import DATASETS, METRICS, MODELS
from mmseg.utils import register_all_modules
from torch.nn import functional as F

import wwtpseg  # noqa: F401 -- registers WWTP dataset/model/metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/v5/rpgv_v5.py'
DEFAULT_CHECKPOINT = ROOT / 'work_dirs/rpgv_v5_1m/best_binary_Foreground_IoU_iter_12000.pth'
MODES = ('full', 'no_geometry', 'no_contour', 'shuffled_depth')


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def make_metrics(cfg: Config, dataset) -> dict:
    # Keep the official 1 m / 1.5 px boundary and 64 m² small-area settings.
    metrics = {
        'iou': METRICS.build(cfg.val_evaluator[0]),
        'binary': METRICS.build(cfg.val_evaluator[1]),
        'topology': METRICS.build(cfg.val_evaluator[2]),
    }
    for metric in metrics.values():
        metric.dataset_meta = dataset.metainfo
    return metrics


def output_logits(model, rgb: torch.Tensor, decoded: torch.Tensor) -> dict:
    return model.refiner(rgb, decoded)


def to_binary_mask(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    logits = F.interpolate(logits.float(), size=size, mode='bilinear', align_corners=False)
    return (logits > 0).to(torch.long).squeeze(0).cpu()


def add_result(metrics: dict, source_sample, mask: torch.Tensor) -> dict:
    # Retain the original dataset sample and metainfo, including img_path and
    # unpadded ground truth. Each metric immediately records scalar results.
    if tuple(mask.shape[-2:]) != tuple(source_sample.gt_sem_seg.data.shape[-2:]):
        raise ValueError('prediction and unpadded validation GT differ in size')
    source_sample.pred_sem_seg = PixelData(data=mask)
    # Evaluator normally performs this conversion before calling IoUMetric.
    payload = source_sample.to_dict()
    for metric in metrics.values():
        metric.process({}, [payload])
    return dict(**metrics['binary'].results[-1], **metrics['topology'].results[-1])


def build_model(cfg: Config, checkpoint: Path, device: torch.device):
    model_cfg = cfg.model.copy()
    model_cfg.rgb_encoder.init_cfg = None
    model = MODELS.build(model_cfg)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'state_dict' not in state:
        raise ValueError('checkpoint has no state_dict')
    model.load_state_dict(state['state_dict'], strict=True)
    iteration = state.get('meta', {}).get('iter')
    del state
    model.to(device).eval()
    return model, iteration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path, required=True,
                        help='New JSON path; existing files are never overwritten')
    parser.add_argument('--limit', type=int, default=0, help='0 means all 304 val images')
    parser.add_argument('--variants', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--amp', action='store_true',
                        help='CUDA autocast diagnostic; default is FP32 like validation')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    args.variants = list(dict.fromkeys(args.variants))
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; pass --device cpu for a CPU diagnostic')
    if args.amp and args.device != 'cuda':
        parser.error('--amp requires --device cuda')
    return args


def main() -> None:
    args = parse_args()
    register_all_modules(init_default_scope=True)
    config_path = DEFAULT_CONFIG.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists():
        raise FileExistsError(f'choose a new output JSON: {output}')
    cfg = Config.fromfile(config_path)
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    if len(dataset) != 304:
        raise ValueError(f'expected the 304-image validation split; found {len(dataset)}')
    count = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))
    device = torch.device(args.device)
    model, checkpoint_iter = build_model(cfg, checkpoint, device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    metrics = {name: make_metrics(cfg, dataset) for name in args.variants}
    shared_target_cache = {}
    for group in metrics.values():
        group['binary']._target_cache = shared_target_cache
    rows = {name: [] for name in args.variants}
    timing = defaultdict(list)
    run_started = time.perf_counter()

    for index in range(count):
        image_started = time.perf_counter()
        item = dataset[index]
        sample = item['data_samples']
        batch = model.data_preprocessor(
            dict(inputs=[item['inputs']], data_samples=[sample]), training=False)
        inputs = batch['inputs']
        if inputs.shape[0] != 1 or inputs.shape[1] != 5:
            raise ValueError(f'expected one five-channel validation image, got {tuple(inputs.shape)}')
        source_size = tuple(sample.gt_sem_seg.data.shape[-2:])
        sync(device)
        timing['data_ms'].append(1000 * (time.perf_counter() - image_started))

        # A single MiT pass supplies all modes. In particular, no_geometry is
        # the same learned decoder without a geometry residual.
        autocast = (torch.autocast('cuda', dtype=torch.float16)
                    if args.amp else nullcontext())
        with torch.inference_mode(), autocast:
            step = time.perf_counter()
            rgb, depth, confidence = model._split_inputs(inputs)
            features = tuple(model.rgb_encoder(rgb))
            sync(device)
            timing['shared_rgb_ms'].append(1000 * (time.perf_counter() - step))

            logits_by_mode = {}
            if 'full' in metrics or 'no_contour' in metrics:
                step = time.perf_counter()
                geometry = model.geometry_encoder(depth, confidence)
                decoded = model.decoder(features, geometry, confidence)
                head = output_logits(model, rgb, decoded)
                if 'full' in metrics:
                    logits_by_mode['full'] = head['final_logits']
                if 'no_contour' in metrics:
                    logits_by_mode['no_contour'] = F.interpolate(
                        head['coarse_logits'], size=head['final_logits'].shape[-2:],
                        mode='bilinear', align_corners=False)
                sync(device)
                timing['full_decode_ms'].append(1000 * (time.perf_counter() - step))

            if 'no_geometry' in metrics:
                step = time.perf_counter()
                decoded = model.decoder(features)
                logits_by_mode['no_geometry'] = output_logits(model, rgb, decoded)['final_logits']
                sync(device)
                timing['no_geometry_decode_ms'].append(1000 * (time.perf_counter() - step))

            if 'shuffled_depth' in metrics:
                step = time.perf_counter()
                displaced_depth = depth.roll(shifts=(257, 193), dims=(-2, -1))
                geometry = model.geometry_encoder(displaced_depth, confidence)
                decoded = model.decoder(features, geometry, confidence)
                logits_by_mode['shuffled_depth'] = output_logits(model, rgb, decoded)['final_logits']
                sync(device)
                timing['shuffled_depth_decode_ms'].append(1000 * (time.perf_counter() - step))

        metric_started = time.perf_counter()
        for name in args.variants:
            mask = to_binary_mask(logits_by_mode[name], source_size)
            result = add_result(metrics[name], sample, mask)
            rows[name].append(dict(index=index, image=Path(sample.metainfo['img_path']).name,
                                   tp=result['tp'], fp=result['fp'], fn=result['fn'],
                                   detached_fp_components=result['detached_fp_components'],
                                   detached_fp_pixels=result['detached_fp_pixels'],
                                   pred_components=result['pred_components'],
                                   pred_holes=result['pred_holes'],
                                   pred_hole_pixels=result['pred_hole_pixels'],
                                   gt_components=result['gt_components'],
                                   gt_holes=result['gt_holes']))
        timing['metric_ms'].append(1000 * (time.perf_counter() - metric_started))
        # Reuse GT EDT across modes, never retain 304 full-resolution EDTs.
        shared_target_cache.clear()
        if (index + 1) % 32 == 0 or index + 1 == count:
            print(f'val {index + 1}/{count}, elapsed {time.perf_counter() - run_started:.1f}s',
                  flush=True)

    aggregate = {}
    for name, group in metrics.items():
        aggregate[name] = {
            'iou': group['iou'].compute_metrics(group['iou'].results),
            'binary': group['binary'].compute_metrics(group['binary'].results),
            'topology': group['topology'].compute_metrics(group['topology'].results),
        }
    elapsed = time.perf_counter() - run_started
    provenance = dict(
        split='val', checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
        checkpoint_iter=checkpoint_iter, config=str(config_path),
        config_sha256=file_sha256(config_path),
        resolved_model_config_sha256=hashlib.sha256(
            cfg.pretty_text.encode('utf-8')).hexdigest(),
        script_sha256=file_sha256(Path(__file__)),
        dataset_root=str(Path(cfg.data_root).resolve()), dataset_images=len(dataset),
        evaluated_images=count, variants=args.variants, amp=args.amp,
        dtype='float16 autocast' if args.amp else 'float32',
        device=str(device), torch_version=torch.__version__,
        cuda_device=(torch.cuda.get_device_name(device) if device.type == 'cuda' else None),
        threshold='interpolated foreground logit > 0',
        shuffled_depth='roll depth by (257,193) pixels, keep Q spatially fixed',
        no_contour='same full geometry decode, use upsampled coarse logits',
        postprocessing=False,
    )
    timing_summary = {
        key: dict(mean_ms=sum(values) / len(values), total_ms=sum(values))
        for key, values in timing.items() if values
    }
    timing_summary['total_wall_s'] = elapsed
    timing_summary['images_per_s'] = count / elapsed
    if device.type == 'cuda':
        timing_summary['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated(device)
    report = dict(provenance=provenance, metrics=aggregate,
                  per_image=rows, timing=timing_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f'.tmp.{os.getpid()}')
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'saved {output}', flush=True)


if __name__ == '__main__':
    main()
