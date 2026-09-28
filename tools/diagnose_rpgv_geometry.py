#!/usr/bin/env python3
"""Paired geometry-on/off diagnostics on fixed WWTP validation windows.

This is a bounded diagnostic, not a replacement for full slide validation.
Reuses each checkpoint's RGB features and global token for both predictions.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
from PIL import Image

import wwtpseg  # noqa: F401


RGB_PREFIXES = ('rgb_encoder.', 'global_film.', 'rgb_aux_head.',
                'rgb_boundary_head.', 'decoder.', 'refiner.')


def checkpoint_state(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    return {k.removeprefix('module.'): v for k, v in data['state_dict'].items()}


def confusion(prob, target):
    pred, truth, valid = prob >= 0.5, target == 1, target != 255
    return dict(tp=int((pred & truth & valid).sum()),
                fp=int((pred & ~truth & valid).sum()),
                fn=int((~pred & truth & valid).sum()))


def aggregate(rows):
    counts = {k: sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn')}
    counts['iou'] = 100 * counts['tp'] / max(sum(counts.values()), 1)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-checkpoint', type=Path, required=True)
    parser.add_argument('--geometry-checkpoint', type=Path, required=True)
    parser.add_argument('--joint-checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--data-root', type=Path, default=Path('wwtp_semantic_dataset'))
    parser.add_argument('--images', type=int, default=6)
    parser.add_argument('--crop-size', type=int, default=1024)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.images < 1 or args.crop_size < 64:
        parser.error('images must be positive and crop-size >=64')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile('configs/experiments/rpgv_v2_stage3_joint.py')
    cfg.model.rgb_encoder.init_cfg = None
    model = MODELS.build(cfg.model).eval()
    reference = checkpoint_state(args.rgb_checkpoint)
    stage2 = checkpoint_state(args.geometry_checkpoint)
    rgb_keys = [k for k in reference if k.startswith(RGB_PREFIXES)]
    identical = all(torch.equal(reference[k], stage2[k]) for k in rgb_keys)
    del stage2
    files = sorted((args.data_root / 'images/val').glob('*.png'))
    selected = [files[i] for i in np.linspace(0, len(files) - 1,
                min(args.images, len(files)), dtype=int)]
    report = dict(scope='fixed native-resolution center windows, shared global scene token',
                  selection='evenly spaced indices over sorted validation filenames',
                  crop_size=args.crop_size, stage2_rgb_bitwise_identical=identical,
                  rgb_checkpoint=str(args.rgb_checkpoint),
                  geometry_checkpoint=str(args.geometry_checkpoint), checkpoints=[])
    for checkpoint in [args.rgb_checkpoint, *args.joint_checkpoints]:
        state = checkpoint_state(checkpoint)
        model.load_state_dict(state, strict=True)
        drifts = {}
        for prefix in RGB_PREFIXES:
            keys = [k for k in rgb_keys if k.startswith(prefix)]
            numerator = sum((state[k].float() - reference[k].float()).square().sum().item() for k in keys)
            denominator = sum(reference[k].float().square().sum().item() for k in keys)
            drifts[prefix] = (numerator / max(denominator, 1e-12)) ** 0.5
        del state
        rows = []
        with torch.inference_mode():
            for path in selected:
                rgb = np.asarray(Image.open(path).convert('RGB')).copy()
                h, w = rgb.shape[:2]
                y, x = max((h - args.crop_size) // 2, 0), max((w - args.crop_size) // 2, 0)
                # Keep the loader's uint16/uint8 archive conventions.
                geo = np.load(args.data_root / 'pseudo_geometry/val' / (path.stem + '.npz'))
                depth, quality = geo['depth'], geo['reliability']
                depth = depth.astype(np.float32) / (65535 if depth.dtype == np.uint16 else 1)
                quality = quality.astype(np.float32) / (255 if quality.dtype == np.uint8 else 1)
                tensor = torch.from_numpy(np.concatenate([
                    rgb[..., ::-1].copy().astype(np.float32), depth[..., None] * 255,
                    quality[..., None] * 255], axis=-1)).permute(2, 0, 1)[None]
                global_img = torch.nn.functional.interpolate(tensor[:, :3], size=(512, 512),
                                   mode='bilinear', align_corners=False)
                token = model._global_token(global_img)
                inputs = tensor[..., y:y + args.crop_size, x:x + args.crop_size]
                out = model._run_network(inputs, global_token=token)
                normalized, _, _ = model._split_inputs(inputs)
                fallback = model._decode_and_refine(normalized, out['rgb_features'],
                                                    torch.zeros_like(out['reliability']))
                predictions = []
                for logits in (out['final_logits'], fallback['final_logits']):
                    predictions.append(torch.nn.functional.interpolate(logits.float(),
                        size=inputs.shape[-2:], mode='bilinear', align_corners=False).sigmoid()[0, 0].numpy())
                gt = np.asarray(Image.open(args.data_root / 'annotations/val' / path.name))
                gt = gt[y:y + args.crop_size, x:x + args.crop_size]
                on, off = predictions
                row = dict(image=path.name, xywh=[x, y, inputs.shape[-1], inputs.shape[-2]],
                           geometry_on=confusion(on, gt), geometry_off=confusion(off, gt),
                           probability_mae=float(np.abs(on - off).mean()),
                           flipped_pixels=int((((on >= .5) != (off >= .5)) & (gt != 255)).sum()),
                           reliability_mean=out['reliability'].mean().item())
                rows.append(row)
                print(checkpoint.name, path.name, row['flipped_pixels'], flush=True)
        record = dict(checkpoint=str(checkpoint), rgb_relative_l2_drift=drifts, per_image=rows,
                      geometry_on=aggregate([r['geometry_on'] for r in rows]),
                      geometry_off=aggregate([r['geometry_off'] for r in rows]))
        report['checkpoints'].append(record)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in record.items() if k != 'per_image'}), flush=True)


if __name__ == '__main__':
    main()
