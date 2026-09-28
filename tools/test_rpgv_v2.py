#!/usr/bin/env python3
"""V2 stage/gradient, fallback, contour, ignore-mask and inference regressions."""
import copy
import unittest
from pathlib import Path

import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401
from initialize_rpgv_v2 import transfer_upstream
from smoke_test import _sample, _add_pseudo_geometry, _check_rpgv_stage_freezing
from wwtpseg.models.segmentors.rpgv_net_v2 import contour_target, region_and_boundary_loss
from wwtpseg.models.utils.rpgv_v2_modules import UnifiedContourHead


def build(stage='joint', v2=True):
    name = {'rgb': 'stage1_rgb', 'geometry': 'stage2_geometry', 'joint': 'stage3_joint'}[stage]
    cfg = Config.fromfile(f'configs/experiments/rpgv_{"v2_" if v2 else ""}{name}.py')
    cfg.model.rgb_encoder.init_cfg = None
    cfg.model.global_thumbnail_size = 64
    cfg.model.geometry_dropout_prob = 0
    return MODELS.build(cfg.model)


class V2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        torch.set_num_threads(2)

    def test_all_configs(self):
        for path in sorted(Path('configs/ablations_v2').glob('*.py')):
            cfg = Config.fromfile(path)
            self.assertEqual(cfg.model.type, 'RPGVNetV2')
            self.assertTrue(cfg.enable_early_stopping)
            self.assertIsNone(cfg.load_from)
            self.assertFalse(cfg.model.learnable_loss_weights)
            self.assertEqual(cfg.custom_hooks[0].type, 'RPGVEarlyStoppingHook')

    def test_losses_preserve_edges_and_detect_islands(self):
        target = torch.zeros(1, 1, 16, 16)
        target[..., 4:12, 4:12] = 1
        valid = torch.ones_like(target, dtype=torch.bool)
        perfect = 20 * (target * 2 - 1)
        good = region_and_boundary_loss(perfect, target, valid)
        bad = perfect.clone()
        bad[..., 1, 1] = 20  # isolated FP
        bad[..., 7, 7] = -20  # hole
        noisy = region_and_boundary_loss(bad, target, valid)
        self.assertGreater(noisy[0].item(), good[0].item())
        blurred = region_and_boundary_loss(perfect * 0, target, valid)
        self.assertGreater(blurred[1].item(), good[1].item())
        x = bad.requires_grad_()
        loss = sum(region_and_boundary_loss(x, target, valid & False))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(x.grad.abs().sum().item(), 0)

    def test_sdf_uniform_and_ignore(self):
        for foreground in (0, 1):
            target = torch.full((1, 1, 32, 32), float(foreground))
            valid = torch.ones_like(target, dtype=torch.bool)
            sdf, known = contour_target(target, valid, (16, 16), 3)
            self.assertTrue(known.all())
            self.assertTrue((sdf == (2 * foreground - 1)).all())
            _, known = contour_target(target, valid & False, (16, 16), 3)
            self.assertFalse(known.any())
        target[..., :, 16:] = 0
        valid[..., 12:20, 12:20] = False
        _, known = contour_target(target, valid, (16, 16), 3)
        self.assertFalse(known[..., 5:11, 5:11].any())

    def test_head_identity_bound_and_odd_size(self):
        head = UnifiedContourHead(8, 8)
        rgb, feature = torch.randn(2, 3, 33, 35), torch.randn(2, 8, 9, 9)
        out = head(rgb, feature)
        self.assertEqual(out['final_logits'].shape[-2:], (17, 18))
        self.assertEqual(out['contour_correction'].abs().sum().item(), 0)
        torch.nn.init.constant_(head.contour[-1].bias, 100)
        out = head(rgb, feature)
        self.assertLessEqual(out['contour_correction'].abs().max().item(), 2)
        self.assertTrue(torch.isfinite(out['final_logits']).all())

    def test_stages_backward_and_strict_transition(self):
        previous = None
        for stage in ('rgb', 'geometry', 'joint'):
            with self.subTest(stage=stage):
                model = build(stage).train()
                if previous is not None:
                    model.load_state_dict(previous, strict=True)
                _check_rpgv_stage_freezing(model, stage)
                image, sample = _sample(64)
                if stage != 'rgb':
                    image = _add_pseudo_geometry(image)
                losses = model.loss(image.unsqueeze(0), [sample])
                loss, _ = model.parse_losses(losses)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
                optimizer.step()
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                previous = copy.deepcopy(model.state_dict())

    def test_zero_reliability_fallback_and_slide(self):
        model = build().eval()
        # Nonzero geometry projections ensure this is not merely a zero-init test.
        torch.nn.init.normal_(model.high_fusion.delta_projection[0].weight, std=0.1)
        torch.nn.init.normal_(model.low_fusion.delta_projection[0].weight, std=0.1)
        image, _ = _sample(64)
        inputs = _add_pseudo_geometry(image).unsqueeze(0)
        inputs[:, 4] = 0
        with torch.no_grad():
            first = model._run_network(inputs)['final_logits']
            inputs[:, 3] = 255 - inputs[:, 3]
            second = model._run_network(inputs)['final_logits']
            fallback = model._run_rgb_branch(inputs)['final_logits']
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(second, fallback, rtol=0, atol=0)
            model.test_cfg = dict(mode='slide', crop_size=(64, 64), stride=(48, 48))
            large = torch.nn.functional.interpolate(inputs, size=(81, 97))
            pred = model.inference(large, [dict(img_shape=(81, 97))])
            self.assertEqual(pred.shape, (1, 2, 81, 97))
            self.assertTrue(torch.isfinite(pred).all())

    def test_transfer_and_parameter_reduction(self):
        v1, v2 = build(v2=False), build()
        new_head = copy.deepcopy(v2.refiner.state_dict())
        keys = transfer_upstream(v2, v1.state_dict())
        self.assertGreater(len(keys), 100)
        for k, tensor in new_head.items():
            torch.testing.assert_close(tensor, v2.refiner.state_dict()[k], rtol=0, atol=0)
        counts = []
        for model in (v1, v2):
            counts.append(dict(total=sum(p.numel() for p in model.parameters()),
                               prediction=sum(p.numel() for m in (model.decoder, model.refiner, model.detail_refiner)
                                              for p in m.parameters())))
        print('\nParameter counts:', counts, flush=True)
        self.assertLess(counts[1]['prediction'], counts[0]['prediction'] / 2)
        self.assertLess(counts[1]['total'], counts[0]['total'])

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA FP16 autocast')
    def test_cuda_amp_updates(self):
        for stage in ('rgb', 'geometry', 'joint'):
            with self.subTest(stage=stage):
                model = build(stage).cuda().train()
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad], lr=1e-4)
                scaler = torch.cuda.amp.GradScaler(init_scale=128)
                image, sample = _sample(128)
                if stage != 'rgb':
                    image = _add_pseudo_geometry(image)
                image, sample = image.cuda(), sample.to('cuda')
                with torch.autocast('cuda', dtype=torch.float16):
                    loss, _ = model.parse_losses(model.loss(image.unsqueeze(0), [sample]))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                scaler.step(optimizer)
                scaler.update()
                self.assertTrue(optimizer.state, 'AMP skipped optimizer update')
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
