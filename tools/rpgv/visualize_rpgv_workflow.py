#!/usr/bin/env python3
"""Visualize one RPGV-Net inference window without changing source data.

This script reads a finished joint-stage checkpoint, one val/test image, its
offline pseudo geometry, and its mask. It writes only to a new output folder.
Depth Anything is not run: D0 and Q0 are the exact archived model inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
from torch.nn import functional as F

register_all_modules()
import wwtpseg  # noqa: F401  # Register RPGVNet and its project modules.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, help='joint-stage or joint ablation config')
    parser.add_argument('checkpoint', type=Path, help='finished checkpoint snapshot')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--image-id', required=True, help='PNG filename stem')
    parser.add_argument('--data-root', type=Path, help='override configured dataset root')
    parser.add_argument('--pseudo-root', type=Path, help='override configured pseudo root')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                        help='one local inference window; default: centered config crop')
    parser.add_argument('--display-zoom', type=int, nargs=4,
                        metavar=('X', 'Y', 'W', 'H'),
                        help='paper figure crop within ROI; inference still uses the full ROI')
    parser.add_argument('--device', default='cuda', help='e.g. cuda:1 or cpu')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='a NEW directory outside data and checkpoint trees')
    parser.add_argument('--save-maps', action='store_true',
                        help='also save full-resolution analysis maps as NPZ')
    parser.add_argument('--correction-limit', type=float, default=0.25,
                        help='symmetric Dhat-D0 color limit for paper figure')
    parser.add_argument('--response-limit', type=float,
                        help='shared feature-response color limit; default: joint 99th percentile')
    parser.add_argument('--paper-dpi', type=int, default=600,
                        help='publication PNG resolution; PDF has vector text')
    return parser.parse_args()


def map_from_archive(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    """Match LoadPseudoGeometry's archive conversion exactly."""
    value = np.asarray(value).squeeze()
    if value.shape != shape:
        raise ValueError(f'{name}: shape {value.shape}, expected {shape}')
    if np.issubdtype(value.dtype, np.integer):
        value = value.astype(np.float32) / np.iinfo(value.dtype).max
    else:
        value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f'{name}: non-finite values')
    return np.clip(value, 0.0, 1.0)


def configured_pseudo_root(cfg: Config, data_root: Path) -> Path:
    for step in cfg.test_dataloader.dataset.pipeline:
        if step.get('type') == 'LoadPseudoGeometry':
            return Path(step['pseudo_root']).expanduser().resolve()
    return (data_root / 'pseudo_geometry').resolve()


def choose_roi(
    requested: list[int] | None,
    image_shape: tuple[int, int],
    crop_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = image_shape
    if requested is None:
        crop_h, crop_w = crop_size
        if crop_h > height or crop_w > width:
            raise ValueError('image is smaller than the configured crop; specify --roi')
        x, y, w, h = (width - crop_w) // 2, (height - crop_h) // 2, crop_w, crop_h
    else:
        x, y, w, h = requested
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError(f'ROI {(x, y, w, h)} is outside image {(width, height)}')
    if h % 32 or w % 32:
        raise ValueError('ROI width and height must be multiples of 32')
    return x, y, w, h


def choose_display_zoom(
    requested: list[int] | None, roi_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    width, height = roi_size
    if requested is None:
        return 0, 0, width, height
    x, y, w, h = requested
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError(
            f'display zoom {(x, y, w, h)} is outside ROI {(width, height)}')
    return x, y, w, h


def check_output_path(output: Path, forbidden: list[Path]) -> None:
    if output.exists():
        raise FileExistsError(f'output directory already exists: {output}')
    for source in forbidden:
        source = source.resolve()
        if output == source or source in output.parents:
            raise ValueError(f'output directory cannot be inside {source}')


def upsample(value: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    value = F.interpolate(value.float(), size=size, mode='bilinear',
                          align_corners=False)
    return value[0, 0].detach().cpu().numpy()


def probability(logits: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    logits = F.interpolate(logits.float(), size=size, mode='bilinear',
                           align_corners=False)
    return logits.sigmoid()[0, 0].detach().cpu().numpy()


def mean_abs_residual(module, inputs, output, maps: dict, name: str) -> None:
    del module
    # The actual fused feature is RGB + Qd * tanh(projected delta), unless
    # reliability_weighting is disabled. Magnitude includes all these effects.
    maps[name] = (output.float() - inputs[0].float()).abs().mean(
        dim=1, keepdim=True).detach()
    maps[name + '_candidate'] = inputs[1].float().abs().mean(
        dim=1, keepdim=True).detach()


def capture_maps(model, inputs: torch.Tensor, global_token: torch.Tensor):
    """Capture genuine intermediate activations during one joint forward."""
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def validation_hook(name):
        def hook(module, args, output):
            del module, args
            # DFGV has a channel-wise sigmoid weight, not a scalar gate.
            captured[name] = output.float().sigmoid().mean(
                dim=1, keepdim=True).detach()
        return hook

    def fusion_hook(name):
        def hook(module, args, output):
            mean_abs_residual(module, args, output, captured, name)
        return hook

    handles.extend([
        model.frequency_validator.high_validator.register_forward_hook(
            validation_hook('boundary_validation_mean')),
        model.frequency_validator.region_validator.register_forward_hook(
            validation_hook('region_validation_mean')),
        model.high_fusion.register_forward_hook(fusion_hook('boundary_response')),
        model.low_fusion.register_forward_hook(fusion_hook('region_response')),
        model.refiner.register_forward_hook(
            lambda module, args, output: captured.update(
                refinement_gate=output['refinement_gate'].detach())),
        model.detail_refiner.register_forward_hook(
            lambda module, args, output: captured.update(
                detail_gate=output['refinement_gate'].detach())),
    ])
    try:
        outputs = model._run_network(inputs, global_token=global_token)
    finally:
        for handle in handles:
            handle.remove()
    return outputs, captured


def make_panels(
    rgb: np.ndarray,
    depth: np.ndarray,
    q0: np.ndarray,
    gt: np.ndarray,
    outputs: dict,
    captured: dict,
    fallback: dict,
    model,
) -> tuple[list[tuple[str, np.ndarray, str, float | None, float | None]], dict[str, np.ndarray]]:
    size = depth.shape
    # RPGVNet.encode_decode interpolates logits before foreground softmax.
    p_rgb_aux = probability(outputs['rgb_logits'], size)
    p_geometry_aux = probability(outputs['geometry_logits'], size)
    p_rgb_fallback = probability(fallback['final_logits'], size)
    p_final = probability(outputs['final_logits'], size)
    d_hat = upsample(outputs['corrected_depth'], size)
    d0_rectifier = upsample(outputs['base_depth'], size)
    q_learn = upsample(outputs['learned_reliability'], size)
    q_task = upsample(outputs['reliability'], size)
    effective_weight = q_task if model.component_enabled('reliability_weighting') else np.ones_like(q_task)
    correction = upsample(outputs['corrected_depth'] - outputs['base_depth'], size)
    effect = p_final - p_rgb_fallback
    pred = (p_final >= 0.5).astype(np.uint8)
    error = np.zeros(size, dtype=np.uint8)
    valid = gt != 255
    error[valid & (pred == 1) & (gt == 1)] = 1  # true positive
    error[valid & (pred == 1) & (gt == 0)] = 2  # false positive
    error[valid & (pred == 0) & (gt == 1)] = 3  # false negative
    error[~valid] = 4

    maps = dict(
        depth_d0=depth, depth_d0_at_rectifier_scale=d0_rectifier,
        reliability_q0=q0, corrected_depth=d_hat,
        depth_correction=correction, learned_reliability=q_learn,
        task_reliability=q_task, effective_fusion_weight=effective_weight,
        rgb_aux_probability=p_rgb_aux, geometry_aux_probability=p_geometry_aux,
        rgb_fallback_probability=p_rgb_fallback, final_probability=p_final,
        final_minus_rgb_fallback=effect, final_mask=pred, gt=gt, error=error,
    )
    for name, tensor in captured.items():
        maps[name] = upsample(tensor, size)

    panels = [
        ('RGB (local input)', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), 'rgb', None, None),
        ('Depth Anything D0 (offline)', depth, 'viridis', 0, 1),
        ('Offline reliability Q0', q0, 'magma', 0, 1),
        ('RGR corrected depth Dhat', d_hat, 'viridis', 0, 1),
        ('RGR correction at 1/4 scale', correction, 'coolwarm', -0.25, 0.25),
        ('Learned attenuation Qlearn', q_learn, 'magma', 0, 1),
        ('Task reliability Qd', q_task, 'magma', 0, 1),
        ('RGB auxiliary P', p_rgb_aux, 'viridis', 0, 1),
        ('Geometry auxiliary P', p_geometry_aux, 'viridis', 0, 1),
    ]
    if 'boundary_validation_mean' in maps:
        panels.append(('Haar boundary W (channel mean)', maps['boundary_validation_mean'], 'magma', 0, 1))
    if 'region_validation_mean' in maps:
        panels.append(('Region W (channel mean)', maps['region_validation_mean'], 'magma', 0, 1))
    if (model.component_enabled('boundary_fusion')
            or model.component_enabled('region_fusion')):
        panels.append(('Effective fusion Q', effective_weight, 'magma', 0, 1))
    candidate_name = ('validated' if model.component_enabled('frequency_validation')
                      else 'direct')
    for key, label in (
        ('boundary_response_candidate', f'|{candidate_name} boundary delta| mean'),
        ('boundary_response', '|F1 fused - RGB| mean'),
        ('region_response_candidate', f'|{candidate_name} region delta| mean'),
        ('region_response', '|F3 fused - RGB| mean'),
    ):
        if key in maps:
            limit = max(float(np.percentile(maps[key], 99)), 1e-6)
            panels.append((label, maps[key], 'inferno', 0, limit))
    if model.component_enabled('boundary_refinement') and 'refinement_gate' in maps:
        panels.append(('Boundary refinement gate', maps['refinement_gate'], 'magma', 0, 1))
    if model.component_enabled('detail_refinement') and 'detail_gate' in maps:
        panels.append(('Detail refinement gate', maps['detail_gate'], 'magma', 0, 1))
    panels.extend([
        ('RGB-only decoder P', p_rgb_fallback, 'viridis', 0, 1),
        ('Final P - RGB-only P', effect, 'coolwarm', -1, 1),
        ('Final foreground P', p_final, 'viridis', 0, 1),
        ('Final mask (P >= 0.5)', pred, 'gray', 0, 1),
        ('Ground truth (255 ignored)', gt, 'gt', None, None),
        ('TP green / FP red / FN blue', error, 'error', None, None),
    ])
    return panels, maps


def save_figure(panels, output: Path, title: str) -> None:
    columns = 4
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(20, 4.8 * rows),
                             constrained_layout=True)
    gt_cmap = ListedColormap(['#202020', '#29b765', '#9c9c9c'])
    gt_norm = BoundaryNorm([-0.5, 0.5, 1.5, 255.5], gt_cmap.N)
    error_cmap = ListedColormap(['#202020', '#29b765', '#f04a4a', '#3188e0', '#9c9c9c'])
    for ax, (label, value, color, vmin, vmax) in zip(axes.flat, panels):
        if color == 'rgb':
            ax.imshow(value)
        elif color == 'gt':
            ax.imshow(value, cmap=gt_cmap, norm=gt_norm, interpolation='nearest')
        elif color == 'error':
            ax.imshow(value, cmap=error_cmap, vmin=0, vmax=4,
                      interpolation='nearest')
        else:
            im = ax.imshow(value, cmap=color, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()
    for ax in list(axes.flat)[len(panels):]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=15)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def paper_response_limit(maps: dict[str, np.ndarray], fixed: float | None) -> float:
    if fixed is not None:
        return fixed
    responses = [maps[name] for name in ('boundary_response', 'region_response')
                 if name in maps]
    return max((float(np.percentile(value, 99)) for value in responses),
               default=0.0) or 1e-6


def save_paper_figure(
    maps: dict[str, np.ndarray],
    rgb: np.ndarray,
    output: Path,
    correction_limit: float,
    response_limit: float,
    dpi: int,
    auto_response_limit: bool,
) -> None:
    """A compact, consistently scaled figure for a two-column paper page."""
    depth_norm = Normalize(0, 1)
    weight_norm = Normalize(0, 1)
    probability_norm = Normalize(0, 1)
    correction_norm = TwoSlopeNorm(
        vcenter=0.0, vmin=-correction_limit, vmax=correction_limit)
    response_norm = Normalize(0, response_limit)
    gt_cmap = ListedColormap(['#202020', '#009E73', '#9C9C9C'])
    gt_norm = BoundaryNorm([-0.5, 0.5, 1.5, 255.5], gt_cmap.N)
    error_cmap = ListedColormap([
        '#202020', '#009E73', '#D55E00', '#0072B2', '#9C9C9C'])
    error_norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), error_cmap.N)

    # Each row follows an actual forward stage. Missing ablation components
    # are shown as disabled, rather than invented as zero-valued weights.
    specs = [
        ('RGB', None, 'rgb', None),
        ('Offline depth $D_0$', 'depth_d0', 'cividis', depth_norm),
        ('Offline reliability $Q_0$', 'reliability_q0', 'magma', weight_norm),
        ('Corrected depth $\\hat D$', 'corrected_depth', 'cividis', depth_norm),
        ('RGR correction $\\Delta D$', 'depth_correction', 'coolwarm', correction_norm),
        ('RGB auxiliary $P_r$', 'rgb_aux_probability', 'viridis', probability_norm),
        ('Geometry auxiliary $P_g$', 'geometry_aux_probability', 'viridis', probability_norm),
        ('Task reliability $Q_d$', 'task_reliability', 'magma', weight_norm),
        ('Haar boundary $\\bar W_h$', 'boundary_validation_mean', 'magma', weight_norm),
        ('Region $\\bar W_l$', 'region_validation_mean', 'magma', weight_norm),
        ('Boundary contribution', 'boundary_response', 'inferno', response_norm),
        ('Region contribution', 'region_response', 'inferno', response_norm),
        ('RGB-only decoder', 'rgb_fallback_probability', 'viridis', probability_norm),
        ('Final prediction', 'final_probability', 'viridis', probability_norm),
        ('Ground truth', 'gt', gt_cmap, gt_norm),
        ('Prediction errors', 'error', error_cmap, error_norm),
    ]
    row_labels = ('INPUT + RGR', 'BRANCH EVIDENCE',
                  'VALIDATION + FUSION', 'PREDICTION + GT')
    with plt.rc_context({
        'font.family': 'DejaVu Sans',
        'font.size': 8,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.linewidth': 0.35,
        'savefig.facecolor': 'white',
    }):
        fig, axes = plt.subplots(4, 4, figsize=(7.2, 8.8), facecolor='white')
        fig.subplots_adjust(left=0.072, right=0.985, top=0.97, bottom=0.14,
                            wspace=0.04, hspace=0.18)
        for index, (label, key, cmap, norm) in enumerate(specs):
            ax = axes.flat[index]
            if cmap == 'rgb':
                ax.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                          interpolation='nearest')
            elif key not in maps:
                ax.set_facecolor('#F3F4F6')
                ax.text(0.5, 0.5, 'Disabled in config', ha='center', va='center',
                        fontsize=7, color='#606975', transform=ax.transAxes)
            else:
                ax.imshow(maps[key], cmap=cmap, norm=norm,
                          interpolation='nearest')
                if key == 'final_probability':
                    foreground = maps['gt'] == 1
                    if foreground.any() and (~foreground).any():
                        ax.contour(foreground.astype(np.uint8), levels=[0.5],
                                   colors=['black'], linewidths=1.1)
                        ax.contour(foreground.astype(np.uint8), levels=[0.5],
                                   colors=['white'], linewidths=0.55)
            ax.set_title(f'({chr(97 + index)})  {label}', fontsize=7.5,
                         loc='left', pad=3, color='#1D2733')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#C8CDD2')
                spine.set_linewidth(0.35)

        for row, label in enumerate(row_labels):
            box = axes[row, 0].get_position()
            fig.text(0.025, (box.y0 + box.y1) / 2, label, rotation=90,
                     ha='center', va='center', fontsize=6.4,
                     color='#4B6074', weight='bold')

        colorbars = [
            ('Relative depth', 'cividis', depth_norm),
            ('Q / validation W', 'magma', weight_norm),
            ('Foreground P', 'viridis', probability_norm),
            ('RGR $\\Delta D$', 'coolwarm', correction_norm),
            ('Feature response', 'inferno', response_norm),
        ]
        for col, (label, cmap, norm) in enumerate(colorbars):
            left = 0.075 + col * 0.183
            cax = fig.add_axes([left, 0.085, 0.158, 0.012])
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                              orientation='horizontal')
            cb.ax.tick_params(labelsize=5.8, length=1.7, pad=1)
            fig.text(left, 0.106, label, fontsize=6.5, color='#273746')
        fig.legend(handles=[
            Patch(facecolor='#009E73', label='TP'),
            Patch(facecolor='#D55E00', label='FP'),
            Patch(facecolor='#0072B2', label='FN'),
            Patch(facecolor='#9C9C9C', label='Ignore'),
        ], loc='lower center', bbox_to_anchor=(0.5, 0.019), ncol=4,
            frameon=False, fontsize=6.7, handlelength=1.0, columnspacing=1.5)
        response_note = ('99th percentile color ceiling' if auto_response_limit
                         else 'fixed color ceiling')
        fig.text(0.5, 0.054,
                 'White contour on (n): GT boundary  |  Feature response: '
                 f'mean absolute change ({response_note})',
                 ha='center', fontsize=6.5, color='#4B6074')
        fig.savefig(output / 'paper_workflow.pdf')
        fig.savefig(output / 'paper_workflow.png', dpi=dpi)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.correction_limit <= 0 or args.paper_dpi <= 0:
        raise ValueError('--correction-limit and --paper-dpi must be positive')
    if args.response_limit is not None and args.response_limit <= 0:
        raise ValueError('--response-limit must be positive')
    if Path(args.image_id).name != args.image_id or not args.image_id:
        raise ValueError('--image-id must be a PNG filename stem without a path')
    config_path = args.config.resolve(strict=True)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    cfg = Config.fromfile(str(config_path))
    if cfg.model.get('type') != 'RPGVNet' or cfg.model.get('training_stage') != 'joint':
        raise ValueError('visualization requires an RPGVNet joint-stage config')
    data_root = (args.data_root or Path(cfg.test_dataloader.dataset.data_root)).expanduser().resolve()
    pseudo_root = (args.pseudo_root or configured_pseudo_root(cfg, data_root)).expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    check_output_path(output, [data_root, pseudo_root, checkpoint_path.parent])

    image_path = data_root / 'images' / args.split / f'{args.image_id}.png'
    gt_path = data_root / 'annotations' / args.split / f'{args.image_id}.png'
    pseudo_path = pseudo_root / args.split / f'{args.image_id}.npz'
    for path in (image_path, gt_path, pseudo_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    full_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    with Image.open(str(gt_path)) as _gt_img:
        gt_full = np.array(_gt_img)
    if full_bgr is None or gt_full is None or gt_full.ndim != 2:
        raise ValueError('image must be BGR PNG and GT must be single-channel PNG')
    shape = full_bgr.shape[:2]
    if gt_full.shape != shape or not np.isin(gt_full, [0, 1, 255]).all():
        raise ValueError('GT shape or values do not match WWTPDataset (0, 1, 255)')
    with np.load(pseudo_path, allow_pickle=False) as archive:
        depth_full = map_from_archive(archive['depth'], shape, 'depth')
        q0_full = map_from_archive(archive['reliability'], shape, 'reliability')

    crop_size = tuple(cfg.model.test_cfg.get('crop_size', (1024, 1024)))
    x, y, w, h = choose_roi(args.roi, shape, crop_size)
    zoom_x, zoom_y, zoom_w, zoom_h = choose_display_zoom(
        args.display_zoom, (w, h))
    rgb = full_bgr[y:y + h, x:x + w].copy()
    depth = depth_full[y:y + h, x:x + w].copy()
    q0 = q0_full[y:y + h, x:x + w].copy()
    gt = gt_full[y:y + h, x:x + w].copy()
    input_array = np.concatenate([
        rgb.astype(np.float32),
        depth[..., None] * 255.0,
        q0[..., None] * 255.0,
    ], axis=2)
    device = torch.device(args.device)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(checkpoint_path), map_location='cpu', strict=True)
    model.to(device).eval()

    with torch.inference_mode():
        inputs = torch.from_numpy(np.ascontiguousarray(input_array.transpose(2, 0, 1)))\
            .unsqueeze(0).to(device)
        global_token = None
        if model.use_global_context:
            # Match RPGVNet.slide_inference: bilinear thumbnail of the whole
            # scene, shared by each local window.
            full_rgb_input = torch.from_numpy(
                np.ascontiguousarray(full_bgr.transpose(2, 0, 1)))\
                .unsqueeze(0).to(device=device, dtype=torch.float32)
            thumbnail = F.interpolate(
                full_rgb_input,
                size=(model.global_thumbnail_size, model.global_thumbnail_size),
                mode='bilinear', align_corners=False)
            global_token = model._global_token(thumbnail)
            del full_rgb_input, thumbnail
        outputs, captured = capture_maps(model, inputs, global_token)
        # This is the model's real RGB-only decoder with zero reliability.
        fallback = model._run_rgb_branch(inputs, global_token=global_token)
        panels, maps = make_panels(
            rgb, depth, q0, gt, outputs, captured, fallback, model)

    for name, value in maps.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise FloatingPointError(f'non-finite visualization map: {name}')

    valid = gt != 255
    pred = maps['final_mask']
    tp = int(((pred == 1) & (gt == 1) & valid).sum())
    fp = int(((pred == 1) & (gt == 0) & valid).sum())
    fn = int(((pred == 0) & (gt == 1) & valid).sum())
    metadata = dict(
        config=str(config_path), checkpoint=str(checkpoint_path),
        image=str(image_path), pseudo_geometry=str(pseudo_path), gt=str(gt_path),
        split=args.split, roi_xywh=[x, y, w, h],
        paper_display_zoom_xywh=[zoom_x, zoom_y, zoom_w, zoom_h],
        components=model.component_cfg,
        available_maps=sorted(maps),
        roi_tp=tp, roi_fp=fp, roi_fn=fn,
        roi_iou=tp / (tp + fp + fn) if tp + fp + fn else None,
        map_means={name: float(maps[name].mean()) for name in (
            'depth_correction', 'task_reliability', 'boundary_response',
            'region_response', 'boundary_validation_mean',
            'region_validation_mean', 'final_minus_rgb_fallback')
            if name in maps},
        paper_color_limits=dict(
            correction=abs(args.correction_limit),
            feature_response_mode=(
                'joint_99th_percentile' if args.response_limit is None else 'fixed'),
            feature_response=paper_response_limit({
                name: value[zoom_y:zoom_y + zoom_h, zoom_x:zoom_x + zoom_w]
                for name, value in maps.items() if value.ndim == 2
            }, args.response_limit)),
        note=('Single local window with a full-scene global token; final P is '
              'the local model output, not stitched whole-image slide inference. '
              'Validation weights are channel means; response maps are mean '
              'absolute changes to fused RGB features. RGB-only decoder uses '
              'the same joint checkpoint with geometry and reliability absent.'),
    )
    output.mkdir(parents=True, exist_ok=False)
    save_figure(panels, output / 'workflow.png',
                f'RPGV-Net workflow | {args.split}/{args.image_id} | ROI {(x, y, w, h)}')
    paper_maps = {
        name: value[zoom_y:zoom_y + zoom_h, zoom_x:zoom_x + zoom_w]
        for name, value in maps.items() if value.ndim == 2
    }
    paper_rgb = rgb[zoom_y:zoom_y + zoom_h, zoom_x:zoom_x + zoom_w]
    save_paper_figure(
        paper_maps, paper_rgb, output, args.correction_limit,
        metadata['paper_color_limits']['feature_response'], args.paper_dpi,
        args.response_limit is None)
    (output / 'metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.save_maps:
        np.savez_compressed(output / 'maps.npz', **maps)
    print(f'Wrote {output / "workflow.png"}')


if __name__ == '__main__':
    main()
