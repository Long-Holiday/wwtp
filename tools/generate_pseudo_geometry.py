#!/usr/bin/env python3
"""Generate globally normalized pseudo depth and reliability for RPGV-Net.

The tool combines a low-resolution whole-image estimate with overlapping
high-resolution tiles.  Each tile is scale/shift aligned to the global map,
then blended with a Hann window.  Flip and scale consistency estimates the
per-pixel reliability stored alongside normalized depth.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class Transform:
    name: str
    scale: float = 1.0

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.name == 'identity':
            return image
        if self.name == 'hflip':
            return np.ascontiguousarray(image[:, ::-1])
        if self.name == 'vflip':
            return np.ascontiguousarray(image[::-1])
        if self.name == 'scale':
            return cv2.resize(
                image, dsize=None, fx=self.scale, fy=self.scale,
                interpolation=cv2.INTER_AREA)
        raise ValueError(f'unknown transform: {self.name}')

    def invert(self, depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        if self.name == 'hflip':
            depth = depth[:, ::-1]
        elif self.name == 'vflip':
            depth = depth[::-1]
        if depth.shape != shape:
            depth = cv2.resize(
                depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(depth)


class DepthAnythingPredictor:
    """Thin Hugging Face Depth Anything inference adapter."""

    def __init__(self, model_name: str, device: str) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as error:
            raise RuntimeError(
                'transformers with Depth Anything support is required') from error
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        values = self.processor(
            images=Image.fromarray(image), return_tensors='pt')
        values = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in values.items()
        }
        prediction = self.model(**values).predicted_depth.unsqueeze(1)
        prediction = torch.nn.functional.interpolate(
            prediction, size=(height, width), mode='bicubic',
            align_corners=False)
        return prediction[0, 0].float().cpu().numpy()


def _positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def _scale_shift_align(
    source: np.ndarray,
    reference: np.ndarray,
    max_samples: int = 100_000,
) -> np.ndarray:
    """Robustly fit ``a * source + b`` to a reference relative-depth map."""
    x = source.reshape(-1).astype(np.float64)
    y = reference.reshape(-1).astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 16:
        return source.astype(np.float32)
    if x.size > max_samples:
        indices = np.linspace(0, x.size - 1, max_samples, dtype=np.int64)
        x, y = x[indices], y[indices]

    design = np.stack([x, np.ones_like(x)], axis=1)
    scale, shift = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = scale * x + shift - y
    median = np.median(residual)
    mad = np.median(np.abs(residual - median)) + 1e-8
    inliers = np.abs(residual - median) < 3.0 * 1.4826 * mad
    if inliers.sum() >= 16:
        scale, shift = np.linalg.lstsq(
            design[inliers], y[inliers], rcond=None)[0]
    if scale <= 0.0:
        scale = np.std(y) / (np.std(x) + 1e-8)
        shift = np.mean(y) - scale * np.mean(x)
    return (scale * source + shift).astype(np.float32)


def _resize_long_side(image: np.ndarray, long_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = long_side / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def predict_tiled(
    image: np.ndarray,
    predictor: DepthAnythingPredictor,
    tile_size: int,
    overlap: float,
    global_size: int,
) -> np.ndarray:
    """Align overlapping local estimates to a global reference and blend."""
    height, width = image.shape[:2]
    thumbnail = _resize_long_side(image, global_size)
    global_depth = predictor(thumbnail)
    global_depth = cv2.resize(
        global_depth, (width, height), interpolation=cv2.INTER_LINEAR)
    stride = max(1, round(tile_size * (1.0 - overlap)))
    y_positions = _positions(height, tile_size, stride)
    x_positions = _positions(width, tile_size, stride)
    weighted_sum = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)

    for y in y_positions:
        for x in x_positions:
            y2, x2 = min(y + tile_size, height), min(x + tile_size, width)
            tile = image[y:y2, x:x2]
            local_depth = predictor(tile)
            reference = global_depth[y:y2, x:x2]
            local_depth = _scale_shift_align(local_depth, reference)
            window_y = np.hanning(max(3, y2 - y))[:y2 - y]
            window_x = np.hanning(max(3, x2 - x))[:x2 - x]
            window = np.maximum(np.outer(window_y, window_x), 0.05)
            weighted_sum[y:y2, x:x2] += local_depth * window
            weight_sum[y:y2, x:x2] += window
    return (weighted_sum / np.maximum(weight_sum, 1e-8)).astype(np.float32)


def normalize_and_reliability(
    estimates: list[np.ndarray], tau: float
) -> tuple[np.ndarray, np.ndarray]:
    base = estimates[0]
    aligned = [base]
    for estimate in estimates[1:]:
        aligned.append(_scale_shift_align(estimate, base))
    lower, upper = np.percentile(base, [2.0, 98.0])
    scale = max(float(upper - lower), 1e-6)
    normalized = [np.clip((value - lower) / scale, 0.0, 1.0) for value in aligned]
    stack = np.stack(normalized, axis=0)
    variance = stack.var(axis=0)
    reliability = np.exp(-variance / max(tau, 1e-8))
    return normalized[0].astype(np.float32), reliability.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'data_root', type=Path,
        help='dataset root containing images/{train,val,test}')
    parser.add_argument(
        '--output-root', type=Path,
        help='default: DATA_ROOT/pseudo_geometry')
    parser.add_argument(
        '--model', default='depth-anything/Depth-Anything-V2-Base-hf')
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--tile-size', type=int, default=1024)
    parser.add_argument('--overlap', type=float, default=0.25)
    parser.add_argument('--global-size', type=int, default=518)
    parser.add_argument('--reliability-tau', type=float, default=0.01)
    parser.add_argument(
        '--consistency', nargs='+',
        choices=['identity', 'hflip', 'vflip', 'scale'],
        default=['identity', 'hflip', 'vflip', 'scale'])
    parser.add_argument('--scale-factor', type=float, default=0.75)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError('--overlap must lie in [0, 1)')
    if args.tile_size <= 0 or args.global_size <= 0:
        raise ValueError('tile and global sizes must be positive')
    if args.scale_factor <= 0.0:
        raise ValueError('--scale-factor must be positive')
    if args.reliability_tau <= 0.0:
        raise ValueError('--reliability-tau must be positive')
    output_root = args.output_root or args.data_root / 'pseudo_geometry'
    predictor = DepthAnythingPredictor(args.model, args.device)
    transform_names = ['identity'] + [
        name for name in args.consistency if name != 'identity']
    transforms = [
        Transform(name, args.scale_factor if name == 'scale' else 1.0)
        for name in transform_names
    ]

    for split in args.splits:
        image_dir = args.data_root / 'images' / split
        image_paths = sorted(image_dir.glob('*.png'))
        if not image_paths:
            raise FileNotFoundError(f'no PNG images found in {image_dir}')
        split_output = output_root / split
        split_output.mkdir(parents=True, exist_ok=True)
        for index, image_path in enumerate(image_paths, start=1):
            output_path = split_output / f'{image_path.stem}.npz'
            if output_path.exists() and not args.overwrite:
                print(f'[{split} {index}/{len(image_paths)}] skip {image_path.name}')
                continue
            with Image.open(image_path) as source_image:
                image = np.asarray(source_image.convert('RGB'))
            estimates = []
            for transform in transforms:
                transformed = transform.apply(image)
                estimate = predict_tiled(
                    transformed, predictor, args.tile_size, args.overlap,
                    args.global_size)
                estimates.append(transform.invert(estimate, image.shape[:2]))
            depth, reliability = normalize_and_reliability(
                estimates, args.reliability_tau)
            temporary = output_path.with_suffix('.tmp.npz')
            np.savez_compressed(
                temporary,
                depth=np.round(depth * 65535.0).astype(np.uint16),
                reliability=np.round(reliability * 255.0).astype(np.uint8))
            temporary.replace(output_path)
            print(f'[{split} {index}/{len(image_paths)}] wrote {output_path}')


if __name__ == '__main__':
    main()
