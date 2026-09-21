#!/usr/bin/env python3
"""RPGV numerical regressions; CUDA tests use real AMP optimizer updates.

Run from the project root: python tools/test_rpgv_numerics.py -v
CPU-only installations skip CUDA tests rather than pretending to test AMP.
"""

from __future__ import annotations

import copy
import logging
import math
import unittest
from contextlib import nullcontext

import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.optim import build_optim_wrapper
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

from smoke_test import (
    EXPERIMENTS,
    _add_pseudo_geometry,
    _check_rpgv_backward_paths,
    _check_rpgv_stage_freezing,
    _disable_pretraining,
    _sample,
)
from wwtpseg.models.segmentors.rpgv_net import RPGVNet, _binary_segmentation_loss
from wwtpseg.models.utils import (
    BoundaryResidualRefiner,
    HighResolutionDetailRefiner,
    binary_entropy_from_logits,
)


class RPGVNumericsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_all_modules(init_default_scope=True)
        MMLogger.get_current_instance().setLevel(logging.ERROR)

    def assert_finite_gradients(self, module: torch.nn.Module) -> None:
        gradients = [
            (name, parameter.grad)
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients, 'no backward gradients')
        for name, gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all(), name)

    def test_entropy_saturation_and_reference(self) -> None:
        # This is the previous failure, not a hypothetical FP16 Dice overflow.
        old_probability = torch.tensor([20.0], dtype=torch.float16).sigmoid()
        old_probability = old_probability.clamp(1e-6, 1.0 - 1e-6)
        old_entropy = -(
            old_probability * old_probability.log()
            + (1 - old_probability) * (1 - old_probability).log())
        self.assertFalse(torch.isfinite(old_entropy).all())

        logits = torch.linspace(-8, 8, 65, requires_grad=True)
        probability = logits.detach().double().sigmoid()
        reference = -(
            probability * probability.log()
            + (1 - probability) * (1 - probability).log()) / math.log(2)
        entropy = binary_entropy_from_logits(logits)
        torch.testing.assert_close(entropy, reference.float())
        torch.testing.assert_close(entropy, entropy.flip(0))
        self.assertEqual(entropy[32].item(), 1.0)
        entropy.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

        saturated = torch.tensor(
            [-65504, -100, -20, 0, 20, 100, 65504],
            dtype=torch.float16, requires_grad=True)
        entropy = RPGVNet._entropy(saturated)
        self.assertEqual(entropy.dtype, torch.float32)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(((entropy >= 0) & (entropy <= 1)).all())
        entropy.sum().backward()
        self.assertTrue(torch.isfinite(saturated.grad).all())

    def test_full_resolution_loss_backward(self) -> None:
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for case in ('foreground', 'background', 'sparse', 'ignored'):
                with self.subTest(device=device, case=case):
                    logits = torch.full(
                        (1, 1, 1024, 1024), 20.0, dtype=torch.float16,
                        device=device, requires_grad=True)
                    target = torch.zeros_like(logits)
                    if case == 'foreground':
                        target.fill_(1)
                    elif case == 'sparse':
                        target[..., 500:505, 500:505] = 1
                    valid = torch.full_like(
                        target, case != 'ignored', dtype=torch.bool)
                    context = (
                        torch.autocast('cuda', dtype=torch.float16)
                        if device == 'cuda' else nullcontext())
                    with context:
                        loss = _binary_segmentation_loss(logits, target, valid)
                    self.assertEqual(loss.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(torch.isfinite(logits.grad).all())
                    if case == 'ignored':
                        self.assertEqual(loss.item(), 0.0)
                        self.assertEqual(logits.grad.abs().sum().item(), 0.0)
                    elif case in ('background', 'sparse'):
                        self.assertGreater(logits.grad.abs().max().item(), 0)

    def test_detail_stage_transition(self) -> None:
        torch.manual_seed(42)
        detail = HighResolutionDetailRefiner(decoder_channels=8, channels=4)
        # Nonzero residuals represent a trained RGB detail head; checking only
        # its zero initialization would miss a random Q-dependent coefficient.
        torch.nn.init.normal_(detail.residual[-1].weight, std=0.02)
        rgb = torch.randn(2, 3, 32, 32)
        decoded = torch.randn(2, 8, 8, 8)
        base = torch.randn(2, 1, 8, 8)
        without = detail(rgb, decoded, base, torch.zeros_like(base))['final']
        with_geometry = detail(rgb, decoded, base, torch.ones_like(base))['final']
        torch.testing.assert_close(without, with_geometry, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA FP16 autocast')
    def test_saturated_refiners_autocast_backward(self) -> None:
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                lowres = BoundaryResidualRefiner(channels=8).cuda()
                detail = HighResolutionDetailRefiner(8, channels=4).cuda()
                torch.nn.init.zeros_(lowres.coarse_head.weight)
                torch.nn.init.constant_(lowres.coarse_head.bias, sign * 20)
                # Exercise gate derivatives as well as zero-init residual heads.
                torch.nn.init.normal_(lowres.refinement[-1].weight, std=0.02)
                torch.nn.init.normal_(detail.residual[-1].weight, std=0.02)
                feature = torch.randn(2, 8, 8, 8, device='cuda')
                reliability = torch.rand(2, 1, 8, 8, device='cuda')
                with torch.autocast('cuda', dtype=torch.float16):
                    coarse = lowres(feature, reliability)
                    outputs = detail(
                        torch.randn(2, 3, 32, 32, device='cuda'), feature,
                        coarse['final'], reliability)
                    self.assertEqual(coarse['coarse'].dtype, torch.float16)
                    for name, value in {**coarse, **outputs}.items():
                        self.assertTrue(torch.isfinite(value).all(), name)
                    loss = outputs['final'].float().square().mean()
                loss.backward()
                self.assert_finite_gradients(lowres)
                self.assert_finite_gradients(detail)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA AmpOptimWrapper')
    def test_all_stages_amp_optimizer_updates(self) -> None:
        for stage in ('rpgv_stage1_rgb', 'rpgv_stage2_geometry', 'rpgv_stage3_joint'):
            with self.subTest(stage=stage):
                self._train_stage(stage)

    def _train_stage(self, name: str) -> None:
        torch.manual_seed(42)
        cfg = Config.fromfile(EXPERIMENTS[name])
        model_cfg = copy.deepcopy(cfg.model)
        _disable_pretraining(model_cfg)
        # Preserve all model widths/depths, loss weights and optimizer settings.
        # Only images/thumbnails are reduced; the loss test above covers 1024.
        model_cfg['global_thumbnail_size'] = 64
        model_cfg['data_preprocessor']['size'] = (64, 64)
        model = revert_sync_batchnorm(MODELS.build(model_cfg)).cuda().train()
        _check_rpgv_stage_freezing(model, model.training_stage)
        wrapper = build_optim_wrapper(model, copy.deepcopy(cfg.optim_wrapper))
        accumulation = cfg.optim_wrapper.accumulative_counts
        wrapper.initialize_count_status(model, 0, 16 * accumulation)
        items = [_sample(64, index) for index in range(2)]
        for index, (image, sample) in enumerate(items):
            sample.global_img = PixelData(data=image.clone())
            if model.training_stage != 'rgb':
                # Clean and corrupted supervision must both reach logit BCE.
                sample.pseudo_validity = PixelData(
                    data=torch.full((1, 64, 64), float(index)))
                items[index] = (_add_pseudo_geometry(image), sample)
        batch = dict(
            inputs=[image for image, _ in items],
            data_samples=[sample for _, sample in items])
        updated = []

        def check_step(optimizer, args, kwargs):
            self.assert_finite_gradients(model)
            _check_rpgv_backward_paths(model, model.training_stage)
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    state = optimizer.state[parameter]
                    self.assertTrue(torch.isfinite(parameter).all())
                    for key in ('exp_avg', 'exp_avg_sq'):
                        if key in state:
                            self.assertTrue(torch.isfinite(state[key]).all())
            updated.append(True)

        handle = wrapper.optimizer.register_step_post_hook(check_step)
        try:
            # Dynamic GradScaler may legitimately skip initial overflow steps.
            # Require TWO actual AdamW updates, not just finite forward losses.
            for _ in range(16 * accumulation):
                log_vars = model.train_step(batch, wrapper)
                self.assertTrue(all(torch.isfinite(v).all() for v in log_vars.values()))
                scale = wrapper.loss_scaler.get_scale()
                self.assertTrue(math.isfinite(scale) and scale > 0)
                if len(updated) >= 2:
                    break
            self.assertEqual(len(updated), 2, f'{name}: no successful AMP updates')
            # Both direct/list losses and the diagnostic stage/name are covered.
            for value in (float('nan'), float('inf')):
                with self.assertRaisesRegex(
                    FloatingPointError, f'{model.training_stage}.*loss_probe'
                ):
                    model.parse_losses(dict(loss_probe=[torch.tensor(value).cuda()]))
            print(f'\n[OK] {name}: 2 AdamW updates, accumulation={accumulation}, '
                  f'AMP scale={scale:g}', flush=True)
        finally:
            handle.remove()


if __name__ == '__main__':
    unittest.main()
