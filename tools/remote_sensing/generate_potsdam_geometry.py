#!/usr/bin/env python3
"""Generate Depth Anything V2 pseudo depth and reliability maps for ISPRS Potsdam.

Processes the 512x512 patches in data/remote_sensing/prepared/potsdam/img_dir/{train,val,test}
using Depth-Anything-V2-Base-hf with multi-view consistency transforms (identity, hflip, vflip, scale).
Stores uint16 normalized depth and uint8 reliability in .npz format under
data/remote_sensing/prepared/potsdam/pseudo_geometry/{train,val,test}.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch


def _scale_shift_align(
    source: np.ndarray,
    reference: np.ndarray,
    max_samples: int = 100_000,
) -> np.ndarray:
    """Robustly fit a * source + b to a reference depth map using inlier LSTSQ."""
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
        scale, shift = np.linalg.lstsq(design[inliers], y[inliers], rcond=None)[0]
    if scale <= 0.0:
        scale = np.std(y) / (np.std(x) + 1e-8)
        shift = np.mean(y) - scale * np.mean(x)
    return (scale * source + shift).astype(np.float32)


def normalize_and_reliability(
    estimates: list[np.ndarray], tau: float = 0.01
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


class BatchedDepthPredictor:
    """Efficient batch inference adapter for Depth Anything V2."""

    def __init__(self, model_name: str, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        print(f"Loading {model_name} onto {device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        if self.is_cuda:
            self.model = self.model.half()
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_batch(self, pil_images: list[Image.Image], target_shape: tuple[int, int]) -> list[np.ndarray]:
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {
            k: v.to(self.device, dtype=torch.float16 if (torch.is_floating_point(v) and self.is_cuda) else None)
            if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }
        predictions = self.model(**inputs).predicted_depth.unsqueeze(1)
        if predictions.shape[-2:] != target_shape:
            predictions = torch.nn.functional.interpolate(
                predictions, size=target_shape, mode="bicubic", align_corners=False
            )
        return [pred[0].float().cpu().numpy() for pred in predictions]


def save_npz(out_path: Path, depth: np.ndarray, reliability: np.ndarray) -> None:
    temp_path = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp_path,
        depth=np.round(depth * 65535.0).astype(np.uint16),
        reliability=np.round(reliability * 255.0).astype(np.uint8),
    )
    temp_path.replace(out_path)


def process_split(
    split: str,
    data_root: Path,
    output_root: Path,
    predictor: BatchedDepthPredictor,
    batch_size: int = 16,
    tau: float = 0.01,
    scale_factor: float = 0.75,
    overwrite: bool = False,
    pool: ThreadPoolExecutor | None = None,
) -> int:
    img_dir = data_root / "img_dir" / split
    split_output = output_root / split
    split_output.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        print(f"[{split}] No PNG images found in {img_dir}")
        return 0

    to_process = [p for p in img_paths if overwrite or not (split_output / f"{p.stem}.npz").is_file()]
    print(f"\n[{split}] Total: {len(img_paths)}, To process: {len(to_process)}, Already done: {len(img_paths) - len(to_process)}")

    if not to_process:
        return 0

    futures = []
    start_time = time.time()
    processed_count = 0

    for i in range(0, len(to_process), batch_size):
        batch_paths = to_process[i : i + batch_size]
        B = len(batch_paths)

        # 1. Load images as numpy arrays
        raw_images = []
        for p in batch_paths:
            with Image.open(p) as img:
                raw_images.append(np.asarray(img.convert("RGB")))

        target_shape = raw_images[0].shape[:2]

        # 2. Prepare 4 transformed variants for each image in batch
        # Transforms: Identity, HFlip, VFlip, Scale
        # Batch predict 4 variants
        t_id = [Image.fromarray(img) for img in raw_images]
        t_hf = [Image.fromarray(np.ascontiguousarray(img[:, ::-1])) for img in raw_images]
        t_vf = [Image.fromarray(np.ascontiguousarray(img[::-1, :])) for img in raw_images]
        t_sc = [
            Image.fromarray(cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA))
            for img in raw_images
        ]

        d_id = predictor.predict_batch(t_id, target_shape)
        d_hf = predictor.predict_batch(t_hf, target_shape)
        d_vf = predictor.predict_batch(t_vf, target_shape)
        d_sc = predictor.predict_batch(t_sc, target_shape)

        # 3. Invert transforms and compute reliability
        for idx in range(B):
            inv_id = d_id[idx]
            inv_hf = np.ascontiguousarray(d_hf[idx][:, ::-1])
            inv_vf = np.ascontiguousarray(d_vf[idx][::-1, :])
            inv_sc = cv2.resize(d_sc[idx], (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

            estimates = [inv_id, inv_hf, inv_vf, inv_sc]
            depth, reliability = normalize_and_reliability(estimates, tau=tau)

            out_file = split_output / f"{batch_paths[idx].stem}.npz"
            if pool is not None:
                futures.append(pool.submit(save_npz, out_file, depth, reliability))
            else:
                save_npz(out_file, depth, reliability)

        processed_count += B
        elapsed = time.time() - start_time
        fps = processed_count / max(elapsed, 1e-4)
        eta_sec = (len(to_process) - processed_count) / max(fps, 1e-4)
        if (i // batch_size) % 10 == 0 or processed_count == len(to_process):
            print(f"[{split} {processed_count}/{len(to_process)}] {fps:.1f} imgs/s, ETA: {eta_sec/60:.1f}m")

    if pool is not None:
        for f in futures:
            f.result()

    return processed_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/remote_sensing/prepared/potsdam"),
        help="Root path of prepared Potsdam dataset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory for pseudo geometry (default: <data-root>/pseudo_geometry)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--model",
        default="depth-anything/Depth-Anything-V2-Base-hf",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--tau", type=float, default=0.01, help="Reliability exponential scaling")
    parser.add_argument("--scale-factor", type=float, default=0.75, help="Scale transform factor")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz files")
    args = parser.parse_args()

    output_root = args.output_root or args.data_root / "pseudo_geometry"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Potsdam Pseudo Geometry Generation")
    print(f"Data root:   {args.data_root}")
    print(f"Output root: {output_root}")
    print(f"Splits:      {args.splits}")
    print(f"Device:      {args.device}")
    print(f"Batch size:  {args.batch_size}")

    predictor = BatchedDepthPredictor(args.model, args.device)

    total_processed = 0
    start_total = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for split in args.splits:
            count = process_split(
                split,
                args.data_root,
                output_root,
                predictor,
                batch_size=args.batch_size,
                tau=args.tau,
                scale_factor=args.scale_factor,
                overwrite=args.overwrite,
                pool=pool,
            )
            total_processed += count

    total_elapsed = time.time() - start_total
    print(f"\n[DONE] Successfully processed {total_processed} images in {total_elapsed/60:.2f} minutes.")


if __name__ == "__main__":
    main()
