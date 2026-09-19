#!/usr/bin/env python3
"""Train a WWTP segmentation experiment with MMEngine."""

import argparse
import os

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='experiment config path')
    parser.add_argument('--work-dir', help='directory for logs and checkpoints')
    parser.add_argument(
        '--resume', nargs='?', type=str, const='auto',
        help='resume from a checkpoint path, or auto-resume if no path is given')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override config values, e.g. train_dataloader.batch_size=2')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not cfg.get('work_dir'):
        model_name = os.path.splitext(os.path.basename(args.config))[0]
        cfg.work_dir = os.path.join('work_dirs', model_name)
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume:
        cfg.resume = True
        cfg.load_from = args.resume
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
