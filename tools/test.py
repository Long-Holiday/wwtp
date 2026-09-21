#!/usr/bin/env python3
"""Evaluate a trained WWTP checkpoint on the configured test split."""

import argparse

import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# Patch Mask2FormerHead.predict for slide mode inference compatibility
try:
    from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
    orig_m2f_predict = Mask2FormerHead.predict

    def _patched_m2f_predict(self, x, batch_img_metas, test_cfg):
        seg_logits = orig_m2f_predict(self, x, batch_img_metas, test_cfg)
        if test_cfg.get('mode', 'whole') == 'slide':
            crop_h, crop_w = test_cfg.get('crop_size', (512, 512))
            if seg_logits.shape[-2:] != (crop_h, crop_w):
                seg_logits = F.interpolate(
                    seg_logits, size=(crop_h, crop_w), mode='bilinear', align_corners=False)
        return seg_logits

    Mask2FormerHead.predict = _patched_m2f_predict
except Exception:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--work-dir')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override config values')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.load_from = args.checkpoint
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not cfg.get('work_dir'):
        import os
        model_name = os.path.splitext(os.path.basename(args.config))[0]
        cfg.work_dir = os.path.join('work_dirs', model_name)
    metrics = Runner.from_cfg(cfg).test()
    if args.work_dir and isinstance(metrics, dict):
        import json
        from pathlib import Path
        import time
        out_dir = Path(args.work_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_path = out_dir / f'{ts}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f'Test metrics saved to {out_path}')



if __name__ == '__main__':
    main()
