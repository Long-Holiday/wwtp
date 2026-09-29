#!/usr/bin/env python3
"""V5.1 migration, spatial-gradient, boundary-band and config checks."""

import unittest

import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401
from smoke_test import _add_pseudo_geometry, _sample


VARIANTS = ('rpgv_v51', 'loss_only', 'spatial_only',
            'v5_continue', 'from_scratch')


def build(variant='rpgv_v51', **overrides):
    cfg = Config.fromfile(f'configs/v51/{variant}.py')
    cfg.model.rgb_encoder.init_cfg = None
    cfg.model.geometry_dropout_prob = 0.0
    cfg.model.update(overrides)
    return MODELS.build(cfg.model)


def sample_batch(size=64):
    image, sample = _sample(size)
    return _add_pseudo_geometry(image).unsqueeze(0), sample


class V51Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        torch.set_num_threads(2)

    def test_configs_use_new_data_and_matched_continuation(self):
        configs = {name: Config.fromfile(f'configs/v51/{name}.py')
                   for name in VARIANTS}
        full = configs['rpgv_v51']
        for name, cfg in configs.items():
            with self.subTest(name=name):
                self.assertEqual(cfg.model.type, 'RPGVNetV51')
                self.assertIn('1m', cfg.data_root)
                self.assertEqual(cfg.model.data_preprocessor.size, (1024, 1024))
                self.assertEqual(cfg.train_dataloader.batch_size, 2)
                self.assertEqual(cfg.optim_wrapper.accumulative_counts, 4)
                self.assertEqual(cfg.train_cfg.max_iters,
                                 20000 if name == 'from_scratch' else 6000)
                for split in ('train', 'val', 'test'):
                    dataset = cfg[f'{split}_dataloader'].dataset
                    self.assertEqual(dataset.data_root, cfg.data_root)
                    self.assertIn('LoadPseudoGeometry',
                                  [step.type for step in dataset.pipeline])
        self.assertEqual(full.model.boundary_mode, 'band')
        self.assertTrue(full.model.spatial_refinement)
        self.assertFalse(configs['loss_only'].model.spatial_refinement)
        self.assertEqual(configs['spatial_only'].model.boundary_mode, 'legacy')
        self.assertFalse(configs['v5_continue'].model.spatial_refinement)
        self.assertEqual(configs['v5_continue'].model.boundary_mode, 'legacy')
        self.assertEqual(configs['v5_continue'].model.region_loss_weight,
                         Config.fromfile('configs/v5/rpgv_v5.py').model.region_loss_weight)
        self.assertEqual(configs['v5_continue'].model.final_boundary_loss_weight,
                         Config.fromfile('configs/v5/rpgv_v5.py').model.final_boundary_loss_weight)
        self.assertEqual(configs['v5_continue'].train_cfg,
                         full.train_cfg)
        self.assertEqual(configs['v5_continue'].param_scheduler,
                         full.param_scheduler)
        self.assertIsNone(configs['from_scratch'].load_from)

    def test_v5_state_migration_is_complete_and_initial_prediction_identical(self):
        v5_cfg = Config.fromfile('configs/v5/rpgv_v5.py')
        v5_cfg.model.rgb_encoder.init_cfg = None
        v5_cfg.model.geometry_dropout_prob = 0.0
        source = MODELS.build(v5_cfg.model).eval()
        target = build().eval()
        old_state = source.state_dict()
        new_state = target.state_dict()
        self.assertTrue(all(key in new_state and new_state[key].shape == value.shape
                            for key, value in old_state.items()))
        incompatible = target.load_state_dict(old_state, strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertEqual(set(incompatible.missing_keys),
                         set(new_state) - set(old_state))
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(key.startswith(('decoder.spatial8.',
                                             'decoder.spatial4.'))
                            for key in incompatible.missing_keys))
        for stage in (target.decoder.spatial8, target.decoder.spatial4):
            self.assertEqual(torch.count_nonzero(stage.output.weight), 0)
            self.assertEqual(torch.count_nonzero(stage.output.bias), 0)
        inputs, _ = sample_batch()
        with torch.no_grad():
            torch.testing.assert_close(target.encode_decode(inputs),
                                       source.encode_decode(inputs), rtol=0, atol=0)
        # The v5 continuation control introduces no new parameter keys.
        continuation = build('v5_continue')
        continuation.load_state_dict(old_state, strict=True)

    def test_spatial_output_learns_then_inner_kernels_receive_gradient(self):
        model = build().eval()
        inputs, sample = sample_batch()
        spatial = (model.decoder.spatial8, model.decoder.spatial4)
        loss = model.loss(inputs, [sample])['loss_final']
        loss.backward()
        for stage in spatial:
            self.assertIsNotNone(stage.output.weight.grad)
            self.assertGreater(stage.output.weight.grad.abs().sum().item(), 0)
            self.assertEqual(stage.local.weight.grad.abs().sum().item(), 0)
        optimizer = torch.optim.AdamW(
            [parameter for stage in spatial for parameter in stage.parameters()],
            lr=1e-3)
        optimizer.step()
        for stage in spatial:
            self.assertGreater(stage.output.weight.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        model.loss(inputs, [sample])['loss_final'].backward()
        for stage in spatial:
            for kernel in (stage.local.weight, stage.strip7[0].weight,
                           stage.strip11[0].weight):
                self.assertIsNotNone(kernel.grad)
                self.assertGreater(kernel.grad.abs().sum().item(), 0)

    def test_boundary_band_extremes_and_ignore_handling(self):
        model = build()
        shape = (1, 1, 16, 16)
        for label_value, valid_value in ((0, 1), (1, 1), (0, 0)):
            with self.subTest(label=label_value, valid=valid_value):
                target = torch.full(shape, label_value, dtype=torch.float32)
                valid = torch.full(shape, valid_value, dtype=torch.bool)
                logits = torch.zeros(shape, requires_grad=True)
                loss = model._boundary_band_loss(logits, target, valid)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(loss.item(), 0)
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
        # Unknown labels alone never create a GT edge or supervised band.
        target = torch.zeros(shape)
        valid = torch.ones(shape, dtype=torch.bool)
        valid[..., 6:10, 6:10] = False
        self.assertEqual(model._boundary_band_loss(torch.zeros(shape), target,
                                                    valid).item(), 0)

    def test_boundary_band_rewards_correct_logits_and_has_saturated_gradient(self):
        model = build()
        target = torch.zeros(1, 1, 16, 16)
        target[..., :, 8:] = 1
        valid = torch.ones_like(target, dtype=torch.bool)
        correct = 4 * (2 * target - 1)
        wrong = -correct
        self.assertLess(model._boundary_band_loss(correct, target, valid),
                        model._boundary_band_loss(wrong, target, valid))
        saturated_wrong = (-20 * (2 * target - 1)).requires_grad_()
        loss = model._boundary_band_loss(saturated_wrong, target, valid)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(saturated_wrong.grad.abs().sum().item(), 0)
        # Ignore pixels near an otherwise valid contour cannot contribute.
        valid[..., :, 7:10] = False
        ignored = model._boundary_band_loss(torch.zeros_like(target), target, valid)
        self.assertEqual(ignored.item(), 0)

    def test_all_variants_loss_backward_update_and_strict_restore(self):
        inputs, sample = sample_batch()
        for name in VARIANTS:
            with self.subTest(name=name):
                model = build(name).train()
                self.assertEqual(hasattr(model.decoder, 'spatial8'),
                                 name not in ('loss_only', 'v5_continue'))
                losses = model.loss(inputs, [sample])
                expected = {'loss_final', 'loss_coarse', 'loss_sdf'}
                if name in ('spatial_only', 'v5_continue'):
                    expected |= {'loss_region_consistency', 'loss_final_boundary'}
                else:
                    expected.add('loss_boundary_band')
                self.assertEqual(set(losses), expected)
                total, _ = model.parse_losses(losses)
                self.assertTrue(torch.isfinite(total))
                total.backward()
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    expected_logits = model.encode_decode(inputs)
                    restored = build(name).eval()
                    restored.load_state_dict(model.state_dict(), strict=True)
                    torch.testing.assert_close(restored.encode_decode(inputs),
                                               expected_logits, rtol=0, atol=0)

    def test_depth_fallback_rgb_capacity_and_odd_prediction(self):
        model = build().eval()
        inputs, _ = sample_batch()
        inputs[:, 4] = 0
        changed = inputs.clone()
        changed[:, 3] = 255 - changed[:, 3]
        with torch.no_grad():
            rgb = model._run_rgb_branch(inputs)['seg_logits']
            for candidate in (inputs[:, :3], inputs, changed):
                torch.testing.assert_close(model._run_network(candidate)['seg_logits'],
                                           rgb, rtol=0, atol=0)
        capacity = build(geometry_input='rgb').eval()
        with torch.no_grad():
            expected = capacity.encode_decode(inputs[:, :3])
            torch.testing.assert_close(capacity.encode_decode(inputs),
                                       expected, rtol=0, atol=0)
            torch.testing.assert_close(capacity.encode_decode(changed),
                                       expected, rtol=0, atol=0)
            odd = torch.rand(1, 5, 65, 67) * 255
            odd[:, 4] = 255
            logits = model.encode_decode(odd)
            self.assertEqual(logits.shape, (1, 2, 65, 67))
            self.assertTrue(torch.isfinite(logits).all())
            predicted = model.predict(odd)
            self.assertEqual(predicted[0].pred_sem_seg.data.shape, (1, 65, 67))


if __name__ == '__main__':
    unittest.main()
