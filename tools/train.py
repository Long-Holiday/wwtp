#!/usr/bin/env python3
"""Train a WWTP segmentation experiment with MMEngine."""

import argparse
import os

import torch.nn.functional as F
import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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

# Patch IterBasedTrainLoop to skip advancing dataloader thousands of steps
# when resuming with InfiniteSampler on a dataset that is infinitely shuffled anyway
try:
    from mmengine.runner.loops import IterBasedTrainLoop

    def _fast_iter_train_loop_run(self):
        self.runner.call_hook('before_train')
        self.runner.call_hook('before_train_epoch')
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()
            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)
            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0
                         or self._iter == self._max_iters)):
                self.runner.val_loop.run()
        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    IterBasedTrainLoop.run = _fast_iter_train_loop_run
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
        '--no-resume-scheduler', action='store_true',
        help='do not resume param scheduler state when resuming training')
    parser.add_argument(
        '--disable-early-stopping', action='store_true',
        help='remove configured early stopping and do not inject a default hook')
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
    required_previous_stage = cfg.pop('required_previous_stage', None)
    if required_previous_stage and not cfg.get('load_from') and not cfg.get('resume'):
        raise RuntimeError(
            f'{required_previous_stage} is required. Use '
            '`scripts/train_rpgv_stages.sh`, set the documented checkpoint '
            'environment variable, or pass --resume for an interrupted stage.')

    # Fixed-budget experiment runners may disable this automatic injection.
    # Explicit custom hooks remain the caller's responsibility.
    custom_hooks = list(cfg.get('custom_hooks', []))
    if args.disable_early_stopping:
        custom_hooks = [
            hook for hook in custom_hooks
            if not (isinstance(hook, dict)
                    and hook.get('type') in ('EarlyStoppingHook', 'RPGVEarlyStoppingHook'))]
        cfg.custom_hooks = custom_hooks
    has_early_stopping = any(
        (isinstance(h, dict) and h.get('type') in ('EarlyStoppingHook', 'RPGVEarlyStoppingHook'))
        for h in custom_hooks)
    enable_early_stopping = cfg.pop('enable_early_stopping', True)
    if enable_early_stopping and not args.disable_early_stopping and not has_early_stopping:
        custom_hooks.append(
            dict(
                type='EarlyStoppingHook',
                monitor='binary/Foreground_IoU',
                rule='greater',
                min_delta=1.0,
                patience=3,
                strict=False))
        cfg.custom_hooks = custom_hooks

    runner = Runner.from_cfg(cfg)
    if args.no_resume_scheduler or not cfg.get('resume_param_scheduler', True):
        from mmengine.runner.checkpoint import find_latest_checkpoint

        def _custom_load_or_resume():
            if runner._has_loaded:
                return None
            resume_from = None
            if runner._resume and runner._load_from is None:
                resume_from = find_latest_checkpoint(runner.work_dir)
            elif runner._resume and runner._load_from is not None:
                resume_from = runner._load_from
            if resume_from is not None:
                runner.resume(resume_from, resume_param_scheduler=False)
                # When resuming with a new param scheduler, restore param group lr
                # to its initial_lr so warmup and decay schedules calculate properly
                for g in runner.optim_wrapper.optimizer.param_groups:
                    if 'initial_lr' in g:
                        g['lr'] = g['initial_lr']
                # Clear historical best metrics from message_hub so extended training
                # independently tracks and saves its own best checkpoint in work-dir
                for k in list(runner.message_hub.runtime_info.keys()):
                    if 'best_score' in k or 'best_ckpt' in k:
                        runner.message_hub.runtime_info.pop(k, None)
                runner._has_loaded = True
            elif runner._load_from is not None:
                runner.load_checkpoint(runner._load_from)
                runner._has_loaded = True

        runner.load_or_resume = _custom_load_or_resume

    runner.train()


if __name__ == '__main__':
    main()
