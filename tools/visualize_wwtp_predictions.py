#!/usr/bin/env python3
"""Select WWTP checkpoints by validation IoU and save val/test comparison PNGs.

Run inside the project Docker image. Only the output directory is written.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRIC = 'binary/Foreground_IoU'
SPECS = {
    'RPGV-Net': ('configs/experiments/rpgv_stage3_joint.py',
                 ['work_dirs/rpgv_staged/stage3_joint',
                  'work_dirs/rpgv_stage3_restart_30k']),
    'RPGV_v2': ('configs/experiments/rpgv_v2_stage3_joint.py',
                ['work_dirs/rpgv_v2_staged/stage3_joint']),
    'RPGV_v4': ('configs/v4/rpgv_v4.py',
                ['work_dirs/rpgv_v4']),
    'DeepLabV3+': ('configs/experiments/deeplabv3plus.py',
                   ['work_dirs/deeplabv3plus', 'work_dirs/deeplabv3plus_extended']),
    'HRNet': ('configs/experiments/hrnet.py',
              ['work_dirs/hrnet', 'work_dirs/hrnet_extended']),
    'Mask2Former': ('configs/experiments/mask2former.py',
                    ['work_dirs/mask2former', 'work_dirs/mask2former_extended']),
    'RS-Mamba': ('configs/experiments/rs_mamba.py',
                 ['work_dirs/rs_mamba', 'work_dirs/rs_mamba_extended']),
    'SegFormer': ('configs/experiments/segformer.py',
                  ['work_dirs/segformer', 'work_dirs/segformer_extended']),
    'SegNeXt': ('configs/experiments/segnext.py',
                ['work_dirs/segnext', 'work_dirs/segnext_extended']),
    'U-Net': ('configs/experiments/unet.py',
              ['work_dirs/unet', 'work_dirs/unet_extended']),
    'UNetFormer': ('configs/experiments/unetformer.py',
                   ['work_dirs/unetformer', 'work_dirs/unetformer_extended']),
    'CBR-Net': ('configs/edge_baselines/cbr_net.py',
                ['work_dirs/edge_baselines/cbr_net']),
    'HD-Net': ('configs/edge_baselines/hd_net.py',
               ['work_dirs/edge_baselines/hd_net']),
}
BEST_LINE = re.compile(
    r'The best checkpoint with ([0-9.]+) binary/Foreground_IoU at (\d+) iter')


def select_models(names: list[str]) -> list[dict]:
    selected = []
    for name in names:
        config, candidates = SPECS[name]
        scored = []
        for relative_dir in candidates:
            directory = ROOT / relative_dir
            for checkpoint in directory.glob('best_binary_Foreground_IoU_iter_*.pth'):
                iteration = int(re.search(r'iter_(\d+)\.pth$', checkpoint.name).group(1))
                scores = []
                for log in directory.glob('*/*.log'):
                    for match in BEST_LINE.finditer(log.read_text(errors='replace')):
                        if int(match.group(2)) == iteration:
                            scores.append(float(match.group(1)))
                if not scores:
                    raise RuntimeError(f'No validation score found for {checkpoint}')
                scored.append((max(scores), checkpoint))
        if not scored:
            raise FileNotFoundError(f'No best checkpoint found for {name}')
        score, checkpoint = max(scored, key=lambda pair: pair[0])
        stat = checkpoint.stat()
        selected.append(dict(name=name, slug=name.lower().replace('+', 'plus').replace(' ', '_'),
                             config=str(ROOT / config), checkpoint=str(checkpoint),
                             validation_foreground_iou=score,
                             checkpoint_size=stat.st_size,
                             checkpoint_mtime_ns=stat.st_mtime_ns))
    return selected


def image_ids(data_root: Path, split: str) -> list[str]:
    images = {path.stem for path in (data_root / 'images' / split).glob('*.png')}
    labels = {path.stem for path in (data_root / 'annotations' / split).glob('*.png')}
    if not images or images != labels:
        raise RuntimeError(f'{split}: image/GT mismatch: {len(images)} versus {len(labels)}')
    return sorted(images)


def atomic_png(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    image.save(temp, format='PNG')
    temp.replace(path)


def atomic_json(value: dict, path: Path) -> None:
    """Replace a JSON file without exposing a partially written manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def update_manifest(manifest: dict, path: Path) -> bool:
    """Allow adding models while protecting all recorded prior selections.

    Existing masks are reusable only when their model name still points to
    exactly the same config, checkpoint and validation score. Removing a
    recorded model or silently changing its checkpoint requires a new output
    directory, because either operation would make the gallery ambiguous.
    """
    if not path.exists():
        atomic_json(manifest, path)
        return True
    previous = json.loads(path.read_text())
    if (previous.get('data_root') != manifest['data_root']
            or previous.get('splits') != manifest['splits']):
        raise RuntimeError(
            'Existing output uses another dataset or split selection; '
            'use another output directory')
    previous_models = {model['name']: model for model in previous['models']}
    current_models = {model['name']: model for model in manifest['models']}
    removed = previous_models.keys() - current_models.keys()
    if removed:
        raise RuntimeError(
            f'Existing output contains models omitted by this run: '
            f'{sorted(removed)}. Run with the full model list or use another '
            'output directory')
    changed = []
    for name, old in previous_models.items():
        if current_models[name] != old:
            changed.append(name)
    if changed:
        raise RuntimeError(
            f'Existing model selection changed for {changed}; use another '
            'output directory to avoid mixing checkpoints')
    manifest_changed = previous['models'] != manifest['models']
    if manifest_changed:
        atomic_json(manifest, path)
    return manifest_changed


def infer(model_spec: dict, split: str, data_root: Path, output: Path,
          expected_ids: list[str], device: str) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmseg.registry import DATASETS, MODELS
    from mmseg.utils import register_all_modules

    import wwtpseg  # noqa: F401 - registers custom models/dataset/transforms
    import wwtpseg.edge_baselines  # noqa: F401
    import mmdet.models  # noqa: F401

    # Keep the same sliding-window compatibility fix used by tools/test.py.
    try:
        from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
        if not getattr(Mask2FormerHead.predict, '_wwtp_slide_compat', False):
            original_predict = Mask2FormerHead.predict

            def predict_for_slide(self, features, image_metas, test_cfg):
                logits = original_predict(self, features, image_metas, test_cfg)
                if test_cfg.get('mode', 'whole') == 'slide':
                    size = tuple(test_cfg.get('crop_size', (512, 512)))
                    if logits.shape[-2:] != size:
                        logits = F.interpolate(logits, size=size, mode='bilinear',
                                               align_corners=False)
                return logits

            predict_for_slide._wwtp_slide_compat = True
            Mask2FormerHead.predict = predict_for_slide
    except ImportError:
        pass

    target = output / 'masks' / split / model_spec['slug']
    missing = {item for item in expected_ids if not (target / (item + '.png')).exists()}
    if not missing:
        print(f'{split} {model_spec["name"]}: complete; skipping', flush=True)
        return
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(model_spec['config'])
    dataset_cfg = cfg[f'{split}_dataloader']['dataset'].copy()
    dataset_cfg['data_root'] = str(data_root)
    # The RPGV transform uses its own pseudo root; bind it to the requested data root.
    for transform in dataset_cfg['pipeline']:
        if transform['type'] == 'LoadPseudoGeometry':
            transform['pseudo_root'] = str(data_root / 'pseudo_geometry')
    dataset = DATASETS.build(dataset_cfg)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, model_spec['checkpoint'], map_location='cpu', strict=True,
                    revise_keys=[(r'^module\.', '')])
    model.cfg = cfg
    model.to(device)
    model.eval()
    seen = set()
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            stem = Path(item['data_samples'].metainfo['img_path']).stem
            if stem not in missing:
                continue
            prediction = model.test_step(dict(inputs=[item['inputs']],
                                                   data_samples=[item['data_samples']]))[0]
            mask = prediction.pred_sem_seg.data.squeeze().cpu().numpy()
            size = Image.open(data_root / 'images' / split / (stem + '.png')).size
            if mask.shape != (size[1], size[0]):
                raise RuntimeError(f'{stem}: prediction shape {mask.shape} != {size[::-1]}')
            if not np.isin(mask, [0, 1]).all():
                raise RuntimeError(f'{stem}: unexpected predicted class values')
            atomic_png(Image.fromarray((mask * 255).astype(np.uint8), mode='L'),
                       target / (stem + '.png'))
            seen.add(stem)
            if len(seen) % 25 == 0 or len(seen) == len(missing):
                print(f'{split} {model_spec["name"]}: {len(seen)}/{len(missing)}', flush=True)
    if seen != missing:
        raise RuntimeError(f'{split} {model_spec["name"]}: {len(missing - seen)} images missing')
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()


def compose(split: str, ids: list[str], data_root: Path, output: Path,
            models: list[dict], panel_width: int,
            overwrite: bool = False) -> None:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font_file = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_file, 24) if Path(font_file).exists() else ImageFont.load_default()
    columns = 4
    gap = 18
    caption_height = 42
    panels = [('Original', None), ('GT overlay', None)] + [
        (model['name'], model['slug']) for model in models]
    count = len(panels)
    for number, stem in enumerate(ids, 1):
        destination = output / 'comparisons' / split / (stem + '.png')
        if destination.exists() and not overwrite:
            continue
        with Image.open(data_root / 'images' / split / (stem + '.png')) as source:
            original = source.convert('RGB')
        with Image.open(data_root / 'annotations' / split / (stem + '.png')) as source:
            gt_indices = np.asarray(source)
        if gt_indices.shape != (original.height, original.width):
            raise RuntimeError(f'{stem}: GT size mismatch')
        height = round(panel_width * original.height / original.width)
        cell_height = height + caption_height
        rows = (count + columns - 1) // columns
        sheet = Image.new('RGB', (columns * panel_width + (columns + 1) * gap,
                                  rows * cell_height + (rows + 1) * gap), 'white')
        draw = ImageDraw.Draw(sheet)
        base = original.resize((panel_width, height), Image.Resampling.LANCZOS)
        valid = Image.fromarray((gt_indices == 1).astype(np.uint8) * 255, mode='L')
        overlay = Image.blend(original, Image.new('RGB', original.size, '#ff4d4d'), 0.45)
        overlay.paste(original, mask=valid.point(lambda value: 255 - value))
        overlay = overlay.resize((panel_width, height), Image.Resampling.LANCZOS)
        for index, (label, slug) in enumerate(panels):
            x = gap + (index % columns) * (panel_width + gap)
            y = gap + (index // columns) * (cell_height + gap)
            if index == 0:
                picture = base
            elif index == 1:
                picture = overlay
            else:
                mask_path = output / 'masks' / split / slug / (stem + '.png')
                if not mask_path.exists():
                    raise FileNotFoundError(mask_path)
                with Image.open(mask_path) as source:
                    if source.size != original.size:
                        raise RuntimeError(f'{stem} {label}: mask size mismatch')
                    picture = source.convert('RGB').resize(
                        (panel_width, height), Image.Resampling.NEAREST)
            sheet.paste(picture, (x, y))
            draw.text((x + panel_width // 2, y + height + 7), label,
                      font=font, fill='#202020', anchor='mt')
        atomic_png(sheet, destination)
        if number % 25 == 0 or number == len(ids):
            print(f'{split} comparison: {number}/{len(ids)}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run', 'compose'])
    parser.add_argument('--data-root', type=Path,
                        default=ROOT / 'wwtp_semantic_dataset')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'work_dirs/wwtp_inference_gallery')
    parser.add_argument('--splits', nargs='+', choices=['val', 'test'],
                        default=['val', 'test'])
    parser.add_argument('--models', nargs='+', choices=list(SPECS),
                        default=list(SPECS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--panel-width', type=int, default=480)
    parser.add_argument('--limit', type=int,
                        help='process only this many images per split (for a smoke run)')
    parser.add_argument('--image-ids', nargs='+',
                        help='specific image stems to process in each selected split')
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    if output == data_root or data_root in output.parents:
        parser.error('output must be outside the dataset')
    if args.panel_width < 64:
        parser.error('--panel-width must be at least 64')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    models = select_models(args.models)
    ids_by_split = {split: image_ids(data_root, split) for split in args.splits}
    if args.image_ids:
        requested = set(args.image_ids)
        for split, ids in ids_by_split.items():
            absent = requested - set(ids)
            if absent:
                parser.error(f'{split}: image IDs not found: {sorted(absent)}')
        ids_by_split = {split: [stem for stem in ids if stem in requested]
                        for split, ids in ids_by_split.items()}
    if args.limit is not None:
        ids_by_split = {split: ids[:args.limit] for split, ids in ids_by_split.items()}
    manifest = dict(metric=METRIC, selection='highest validation Foreground IoU',
                    data_root=str(data_root), models=models,
                    splits={split: len(ids) for split, ids in ids_by_split.items()})
    if args.command == 'plan':
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    update_manifest(manifest, manifest_path)
    if args.command == 'run':
        atomic_json(
            dict(status='in_progress', models=len(models),
                 splits=manifest['splits']),
            output / 'completed.json')
        for model in models:
            for split, ids in ids_by_split.items():
                infer(model, split, data_root, output, ids, args.device)
    for split, ids in ids_by_split.items():
        # A run always rebuilds the sheets after all masks are present. This
        # makes an interrupted incremental model addition safe to resume: old
        # model masks are skipped, and stale comparison PNGs are overwritten.
        compose(split, ids, data_root, output, models, args.panel_width,
                overwrite=args.command == 'run')
    if args.command == 'run':
        for split, ids in ids_by_split.items():
            for stem in ids:
                if not (output / 'comparisons' / split / (stem + '.png')).exists():
                    raise RuntimeError(f'{split}/{stem}: missing comparison image')
                for model in models:
                    if not (output / 'masks' / split / model['slug'] /
                            (stem + '.png')).exists():
                        raise RuntimeError(f'{split}/{stem}: missing {model["name"]} mask')
        completion = dict(status='complete', models=len(models),
                          splits=manifest['splits'],
                          mask_pngs=len(models) * sum(manifest['splits'].values()),
                          comparison_pngs=sum(manifest['splits'].values()))
        atomic_json(completion, output / 'completed.json')


if __name__ == '__main__':
    main()
