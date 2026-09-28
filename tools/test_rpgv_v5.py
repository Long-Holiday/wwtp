#!/usr/bin/env python3
"""V5 full-scene configuration, fallback, routing and prediction regression."""

import unittest
from pathlib import Path

import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401
from smoke_test import _add_pseudo_geometry, _sample


VARIANTS = (
    'rpgv_v5', 'no_geometry', 'no_context', 'no_contour',
    'no_structure_losses', 'rgb_capacity_control',
)


def build(variant='rpgv_v5', **overrides):
    cfg = Config.fromfile(f'configs/v5/{variant}.py')
    cfg.model.rgb_encoder.init_cfg = None
    cfg.model.geometry_dropout_prob = 0.0
    cfg.model.update(overrides)
    return MODELS.build(cfg.model)


def grad_sum(module):
    return sum(parameter.grad.abs().sum().item()
               for parameter in module.parameters() if parameter.grad is not None)


class V5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        torch.set_num_threads(2)

    def test_configs_use_converted_full_scenes(self):
        self.assertTrue(set(VARIANTS).issubset(
            {p.stem for p in Path('configs/v5').glob('*.py')}))
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = Config.fromfile(f'configs/v5/{variant}.py')
                self.assertEqual(cfg.model.type, 'RPGVNetV5')
                self.assertIsNone(cfg.load_from)
                self.assertNotIn('training_stage', cfg.model)
                self.assertFalse(cfg.enable_early_stopping)
                self.assertEqual(cfg.train_cfg.max_iters, 20000)
                self.assertEqual(cfg.optim_wrapper.accumulative_counts, 4)
                self.assertEqual(cfg.train_dataloader.batch_size, 2)
                self.assertEqual(cfg.model.data_preprocessor.size, (1024, 1024))
                for split in ('train', 'val', 'test'):
                    dataset = cfg[f'{split}_dataloader'].dataset
                    self.assertEqual(dataset.data_root, cfg.data_root)
                    self.assertIn('1m', dataset.data_root)
                    steps = [step['type'] for step in dataset.pipeline]
                    self.assertIn('LoadPseudoGeometry', steps)
                    self.assertFalse({'RandomCrop', 'Resize', 'RandomResize'} & set(steps))
                self.assertEqual(cfg.model.use_geometry, variant != 'no_geometry')
                self.assertEqual(cfg.model.use_context, variant != 'no_context')
                self.assertEqual(cfg.model.use_contour, variant != 'no_contour')
                self.assertEqual(cfg.model.geometry_input,
                                 'rgb' if variant == 'rgb_capacity_control' else 'depth')

    def test_variants_forward_backward_and_strict_reload(self):
        image, sample = _sample(64)
        inputs = _add_pseudo_geometry(image).unsqueeze(0)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = build(variant).train()
                self.assertEqual(hasattr(model, 'geometry_encoder'),
                                 variant != 'no_geometry')
                self.assertEqual(hasattr(model.decoder, 'context_projections'),
                                 variant != 'no_context')
                self.assertEqual(hasattr(model.decoder, 'fuse4'),
                                 variant != 'no_geometry')
                outputs = model._run_network(inputs)
                self.assertEqual(outputs['seg_logits'].shape, (1, 2, 32, 32))
                losses = model.loss(inputs, [sample])
                expected_losses = {
                    'loss_final', 'loss_coarse', 'loss_region_consistency',
                    'loss_final_boundary', 'loss_sdf',
                }
                if variant == 'no_structure_losses':
                    expected_losses -= {'loss_region_consistency',
                                        'loss_final_boundary'}
                self.assertEqual(set(losses), expected_losses)
                total, _ = model.parse_losses(losses)
                self.assertTrue(torch.isfinite(total))
                total.backward()
                for name, parameter in model.named_parameters():
                    self.assertTrue(parameter.requires_grad, name)
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                torch.optim.AdamW(model.parameters(), lr=1e-4).step()
                model.eval()
                with torch.no_grad():
                    expected = model.encode_decode(inputs)
                    restored = build(variant).eval()
                    restored.load_state_dict(model.state_dict(), strict=True)
                    torch.testing.assert_close(restored.encode_decode(inputs), expected,
                                               rtol=0, atol=0)

    def test_rgb_encoder_is_independent_of_depth(self):
        model = build().eval()
        image, _ = _sample(64)
        inputs = _add_pseudo_geometry(image).unsqueeze(0)
        changed = inputs.clone()
        changed[:, 3] = 255 - inputs[:, 3]
        changed[:, 4] = 255 - inputs[:, 4]
        with torch.no_grad():
            original = model.extract_feat(inputs)
            altered = model.extract_feat(changed)
            reference = model.rgb_encoder(model._normalize_bgr(inputs[:, :3]))
        self.assertEqual(len(original), 4)
        for a, b, c in zip(original, altered, reference):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(a, c, rtol=0, atol=0)

    def test_depth_three_channel_and_zero_confidence_fallback(self):
        model = build().eval()
        image, _ = _sample(64)
        inputs = _add_pseudo_geometry(image).unsqueeze(0)
        inputs[:, 4] = 0
        changed = inputs.clone()
        changed[:, 3] = 255 - inputs[:, 3]
        with torch.no_grad():
            rgb = model._run_rgb_branch(inputs)['seg_logits']
            for candidate in (inputs[:, :3], inputs, changed):
                torch.testing.assert_close(model._run_network(candidate)['seg_logits'],
                                           rgb, rtol=0, atol=0)
            torch.testing.assert_close(model.encode_decode(inputs[:, :3]),
                                       model.encode_decode(inputs), rtol=0, atol=0)

    def test_invalid_depth_cannot_leak_into_valid_neighbors(self):
        model = build().eval()
        image, _ = _sample(64)
        inputs = _add_pseudo_geometry(image).unsqueeze(0)
        inputs[:, 4, :, 32:] = 0
        changed = inputs.clone()
        changed[:, 3, :, 32:] = 255 - inputs[:, 3, :, 32:]
        with torch.no_grad():
            torch.testing.assert_close(model.encode_decode(inputs),
                                       model.encode_decode(changed), rtol=0, atol=0)

    def test_rgb_capacity_control_ignores_both_geometry_channels(self):
        model = build('rgb_capacity_control').eval()
        image, _ = _sample(64)
        inputs = _add_pseudo_geometry(image).unsqueeze(0)
        changed = inputs.clone()
        changed[:, 3] = 255 - inputs[:, 3]
        changed[:, 4] = 0
        with torch.no_grad():
            expected = model.encode_decode(inputs[:, :3])
            torch.testing.assert_close(model.encode_decode(inputs), expected,
                                       rtol=0, atol=0)
            torch.testing.assert_close(model.encode_decode(changed), expected,
                                       rtol=0, atol=0)

    def test_rgb_capacity_control_sample_dropout(self):
        model = build('rgb_capacity_control', geometry_dropout_prob=1.0).train()
        # Keep the capacity control in training mode while disabling the MiT
        # stochastic depth that would otherwise differ between two forwards.
        model.rgb_encoder.eval()
        image, _ = _sample(64)
        inputs = _add_pseudo_geometry(image).unsqueeze(0)
        changed = inputs.clone()
        changed[:, 3] = 255 - inputs[:, 3]
        changed[:, 4] = 0
        with torch.no_grad():
            reference = model._run_rgb_branch(inputs)['seg_logits']
            torch.testing.assert_close(model._run_network(inputs)['seg_logits'],
                                       reference, rtol=0, atol=0)
            torch.testing.assert_close(model._run_network(changed)['seg_logits'],
                                       reference, rtol=0, atol=0)

    def test_v2_v4_full_scene_controls_build_and_loss(self):
        image, sample = _sample(64)
        inputs = _add_pseudo_geometry(image).unsqueeze(0)
        for version in ('v2', 'v4'):
            with self.subTest(version=version):
                cfg = Config.fromfile(f'configs/v5/controls/{version}_1m.py')
                self.assertEqual(cfg.model.type, f'RPGVNet{version.upper()}')
                self.assertEqual(cfg.train_cfg.max_iters, 20000)
                self.assertIn('1m', cfg.data_root)
                cfg.model.rgb_encoder.init_cfg = None
                cfg.model.geometry_dropout_prob = 0.0
                model = MODELS.build(cfg.model).eval()
                with torch.no_grad():
                    logits = model.encode_decode(inputs)
                    self.assertEqual(logits.shape, (1, 2, 64, 64))
                    self.assertTrue(torch.isfinite(logits).all())
                    losses = model.loss(inputs, [sample])
                    self.assertTrue(losses)
                    for name, value in losses.items():
                        if 'loss' in name:
                            self.assertTrue(torch.isfinite(value).all(), name)

    def test_final_loss_reaches_both_geometry_scales(self):
        model = build().eval()
        image, sample = _sample(64)
        model.loss(_add_pseudo_geometry(image).unsqueeze(0), [sample])[
            'loss_final'].backward()
        for name, module in (
            ('geometry stem', model.geometry_encoder.stem),
            ('geometry eighth', model.geometry_encoder.down),
            ('fusion eighth', model.decoder.fuse8),
            ('fusion quarter', model.decoder.fuse4),
        ):
            self.assertGreater(grad_sum(module), 0, name)
        for name, parameter in (
            ('fusion eighth projection', model.decoder.fuse8.output.weight),
            ('fusion quarter projection', model.decoder.fuse4.output.weight),
        ):
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_ignore_and_empty_foreground_are_finite(self):
        model = build().train()
        image, sample = _sample(64)
        inputs = _add_pseudo_geometry(image).unsqueeze(0)
        for value in (0, 255):
            with self.subTest(label=value):
                sample.gt_sem_seg.data.fill_(value)
                model.zero_grad(set_to_none=True)
                losses = model.loss(inputs, [sample])
                total, _ = model.parse_losses(losses)
                self.assertTrue(torch.isfinite(total))
                total.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_odd_size_and_prediction_interface(self):
        model = build().eval()
        inputs = torch.rand(1, 5, 65, 67) * 255
        inputs[:, 4] = 255
        with torch.no_grad():
            logits = model.encode_decode(inputs)
            self.assertEqual(logits.shape, (1, 2, 65, 67))
            self.assertTrue(torch.isfinite(logits).all())
            prediction = model.predict(inputs)
            self.assertEqual(len(prediction), 1)
            self.assertEqual(prediction[0].pred_sem_seg.data.shape, (1, 65, 67))


if __name__ == '__main__':
    unittest.main()
