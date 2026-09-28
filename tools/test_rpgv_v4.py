#!/usr/bin/env python3
"""V4 single-stage integration, gradient routing, ablation and fallback checks."""
import os
import unittest
from pathlib import Path

import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401
from smoke_test import _sample, _add_pseudo_geometry


def build(config='rpgv_v4', **overrides):
    cfg = Config.fromfile(f'configs/v4/{config}.py')
    cfg.model.rgb_encoder.init_cfg = None
    cfg.model.global_thumbnail_size = 64
    cfg.model.geometry_dropout_prob = 0.0
    cfg.model.update(overrides)
    return MODELS.build(cfg.model)


class V4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        torch.set_num_threads(2)

    def test_configs_single_stage(self):
        for path in Path('configs/v4').glob('*.py'):
            cfg = Config.fromfile(path)
            self.assertIsNone(cfg.load_from)
            self.assertNotIn('required_previous_stage', cfg)
            self.assertNotIn('training_stage', cfg.model)
            self.assertEqual(cfg.train_cfg.max_iters, 40000)
            self.assertFalse(cfg.enable_early_stopping)
            types = [step['type'] for step in cfg.train_dataloader.dataset.pipeline]
            self.assertNotIn('RandomPseudoGeometryCorruption', types)
            self.assertEqual('GenerateGlobalThumbnail' in types, cfg.model.use_global_context)

    def test_all_variants_train_and_restore(self):
        for variant in ('rpgv_v4', 'no_frequency', 'no_geometry', 'with_global_context'):
            with self.subTest(variant=variant):
                model = build(variant).train()
                for removed in ('rectifier', 'geometry_encoder', 'geometry_pyramid',
                                'geometry_aux_head', 'rgb_aux_head', 'rgb_boundary_head',
                                'frequency_validator', 'high_fusion', 'low_fusion'):
                    self.assertFalse(hasattr(model, removed), removed)
                image, sample = _sample(64)
                inputs = _add_pseudo_geometry(image).unsqueeze(0)
                losses = model.loss(inputs, [sample])
                self.assertEqual(set(losses), {'loss_final', 'loss_coarse', 'loss_region_consistency',
                                               'loss_final_boundary', 'loss_sdf'})
                total, _ = model.parse_losses(losses)
                total.backward()
                for name, parameter in model.named_parameters():
                    self.assertTrue(parameter.requires_grad, name)
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    expected = model.encode_decode(inputs)
                    restored = build(variant).eval()
                    restored.load_state_dict(model.state_dict(), strict=True)
                    torch.testing.assert_close(restored.encode_decode(inputs), expected, rtol=0, atol=0)

    def test_final_loss_reaches_geometry_and_frequency_immediately(self):
        model = build().eval()
        image, sample = _sample(64)
        model.loss(_add_pseudo_geometry(image).unsqueeze(0), [sample])['loss_final'].backward()
        for i in range(4):
            modules = [model.geometry_downsamples[i], model.geometry_fusions[i]]
            if i < 2:
                modules.append(model.geometry_fusions[i].frequency_projection)
            for module in modules:
                norm = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
                self.assertGreater(norm, 0, f'no final-loss gradient at stage {i}')

    def test_geometry_changes_deeper_semantics(self):
        model = build().eval()
        inputs = torch.rand(1, 5, 65, 67) * 255
        inputs[:, 4] = 255
        alternative = inputs.clone()
        alternative[:, 3] = 255 - inputs[:, 3]
        with torch.no_grad():
            a, b = model.extract_feat(inputs), model.extract_feat(alternative)
        for i in range(4):
            self.assertGreater((a[i] - b[i]).abs().max().item(), 0, f'stage {i}')

    def test_no_geometry_preserves_mit_forward(self):
        model = build('no_geometry').eval()
        inputs = torch.rand(1, 3, 65, 67) * 255
        with torch.no_grad():
            reference = model.rgb_encoder(model._normalize_bgr(inputs))
            actual = model.extract_feat(inputs)
        for expected, result in zip(reference, actual):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_zero_and_partial_confidence_fallback(self):
        model = build().eval()
        inputs = torch.rand(1, 5, 65, 67) * 255
        inputs[:, 4] = 0
        changed = inputs.clone()
        changed[:, 3] = 255 - inputs[:, 3]
        with torch.no_grad():
            expected = model.encode_decode(inputs[:, :3])
            torch.testing.assert_close(model.encode_decode(inputs), expected, rtol=0, atol=0)
            torch.testing.assert_close(model.encode_decode(changed), expected, rtol=0, atol=0)
            fallback = model._run_rgb_branch(changed)['seg_logits']
            full = model._run_network(inputs)['seg_logits']
            torch.testing.assert_close(fallback, full, rtol=0, atol=0)
            # Unknown depth cannot contaminate adjacent valid pixels.
            inputs[:, 4, :, :30] = 255
            changed = inputs.clone()
            changed[:, 3, :, 30:] = 255 - inputs[:, 3, :, 30:]
            torch.testing.assert_close(model.encode_decode(inputs), model.encode_decode(changed), rtol=0, atol=0)
            model.test_cfg = dict(mode='slide', crop_size=(64, 64), stride=(48, 48))
            large = torch.nn.functional.interpolate(inputs, size=(81, 97))
            output = model.inference(large, [dict(img_shape=(81, 97))])
            self.assertEqual(output.shape, (1, 2, 81, 97))
            self.assertTrue(torch.isfinite(output).all())

    def test_ignore_and_geometry_dropout(self):
        model = build(geometry_dropout_prob=1.0).train()
        image, sample = _sample(64)
        sample.gt_sem_seg.data.fill_(255)
        inputs = _add_pseudo_geometry(image).unsqueeze(0)
        original = inputs.clone()
        dropped = model._apply_geometry_dropout(inputs)
        self.assertEqual(dropped[:, 4].abs().sum(), 0)
        torch.testing.assert_close(inputs, original)
        losses = model.loss(inputs, [sample])
        total, _ = model.parse_losses(losses)
        self.assertEqual(total.item(), 0)
        total.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_parameter_counts(self):
        cfg = Config.fromfile('configs/experiments/rpgv_v2_stage3_joint.py')
        cfg.model.rgb_encoder.init_cfg = None
        v2 = MODELS.build(cfg.model)
        v4 = build()
        counts = {name: sum(p.numel() for p in model.parameters())
                  for name, model in [('v2', v2), ('v4', v4)]}
        print('\nParameter counts:', counts, flush=True)
        self.assertLess(counts['v4'], counts['v2'])

    @unittest.skipUnless(os.getenv('RPGV_V4_TEST_CUDA') == '1' and torch.cuda.is_available(),
                         'opt in with RPGV_V4_TEST_CUDA=1 on an available GPU')
    def test_cuda_amp_update(self):
        model = build().cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(init_scale=128)
        image, sample = _sample(128)
        inputs = _add_pseudo_geometry(image).unsqueeze(0).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            total, _ = model.parse_losses(model.loss(inputs, [sample.to('cuda')]))
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        scaler.step(optimizer)
        scaler.update()
        self.assertTrue(optimizer.state, 'AMP skipped the update')
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
