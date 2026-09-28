#!/usr/bin/env python3
"""Verify RGB anchoring, geometry dependence, transfer and real CPU updates."""
import copy
import unittest
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
from mmengine.structures import PixelData

import wwtpseg  # noqa: F401
from initialize_rpgv_v3 import initialize_from_rgb
from smoke_test import _add_pseudo_geometry, _sample


def build_v2():
    cfg = Config.fromfile('configs/experiments/rpgv_v2_stage1_rgb.py')
    cfg.model.rgb_encoder.init_cfg = None
    cfg.model.global_thumbnail_size = 64
    cfg.model.data_preprocessor.size = (64, 64)
    return MODELS.build(cfg.model).eval()


def build_v3(source, **overrides):
    cfg = Config.fromfile('configs/v3/rpgv_v3_adapter.py')
    cfg.model.global_thumbnail_size = 64
    cfg.model.data_preprocessor.size = (64, 64)
    cfg.model.geometry_dropout_prob = 0
    cfg.model.update(overrides)
    model = MODELS.build(cfg.model)
    if source is not None:
        initialize_from_rgb(model, source.state_dict())
    return model


class V3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.source = build_v2()

    def test_configs_and_parameter_count(self):
        for path in Path('configs/v3').rglob('*.py'):
            cfg = Config.fromfile(path)
            self.assertEqual(cfg.model.type, 'RPGVNetV3')
            self.assertFalse(cfg.enable_early_stopping)
        model = build_v3(self.source)
        for name in model.REMOVED_MODULES:
            self.assertFalse(hasattr(model, name), name)
        count = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nv3 parameters: total={count}, trainable={trainable}', flush=True)
        self.assertLess(trainable, 100000)

    def test_reject_random_or_incompatible_anchor(self):
        model = build_v3(None).eval()
        x = torch.zeros(1, 5, 64, 64)
        with self.assertRaisesRegex(RuntimeError, 'verified RGB anchor'):
            model._run_network(x)
        state = dict(self.source.state_dict())
        del state['refiner.coarse_head.weight']
        with self.assertRaisesRegex(ValueError, 'incompatible RGB anchor'):
            initialize_from_rgb(model, state)

    def test_initial_identity_and_exact_fallback(self):
        model = build_v3(self.source).eval()
        image, _ = _sample(64)
        inputs = _add_pseudo_geometry(image)[None]
        with torch.no_grad():
            expected = self.source._run_rgb_branch(image[None])['final_logits']
            out = model._run_network(inputs)
            torch.testing.assert_close(out['final_logits'], expected, rtol=0, atol=0)
            torch.nn.init.normal_(model.geometry_adapter.output.weight, std=.2)
            out = model._run_network(inputs)
            self.assertGreater(out['geometry_correction'].abs().sum().item(), 0)
            self.assertLessEqual(out['geometry_correction'].abs().max().item(), 3)
            inputs[:, 4] = 0
            first = model._run_network(inputs)['final_logits']
            inputs[:, 3] = torch.rand_like(inputs[:, 3]) * 255
            second = model._run_network(inputs)['final_logits']
            torch.testing.assert_close(first, expected, rtol=0, atol=0)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            inputs[:, 4] = 255
            model.correction_strength = 0
            torch.testing.assert_close(model._run_network(inputs)['final_logits'], expected, rtol=0, atol=0)

    def test_geometry_and_rgb_control(self):
        image, _ = _sample(64)
        inputs = _add_pseudo_geometry(image)[None]
        changed = inputs.clone()
        changed[:, 3] = torch.rand_like(changed[:, 3]) * 255
        for enabled in (True, False):
            model = build_v3(self.source, use_geometry_evidence=enabled).eval()
            torch.nn.init.normal_(model.geometry_adapter.output.weight, std=.2)
            with torch.no_grad():
                a = model._run_network(inputs)['final_logits']
                b = model._run_network(changed)['final_logits']
            if enabled:
                self.assertGreater((a - b).abs().sum().item(), 0)
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                changed[:, 4] = 0
                with torch.no_grad():
                    c = model._run_network(changed)['final_logits']
                torch.testing.assert_close(a, c, rtol=0, atol=0)

    def test_optimizer_updates_keep_anchor_bitwise_fixed(self):
        model = build_v3(self.source).train()
        before = copy.deepcopy(model.state_dict())
        for name in model.RGB_MODULES:
            self.assertFalse(getattr(model, name).training, name)
        optimizer = build_optim_wrapper(model, dict(type='OptimWrapper',
            optimizer=dict(type='AdamW', lr=2e-4), accumulative_counts=2))
        image, sample = _sample(64)
        sample.global_img = PixelData(data=image.clone())
        batch = dict(inputs=[_add_pseudo_geometry(image)], data_samples=[sample])
        for _ in range(4):
            values = model.train_step(batch, optimizer)
            self.assertTrue(all(torch.isfinite(v).all() for v in values.values()))
        for name, p in model.named_parameters():
            if name.startswith('geometry_adapter.'):
                continue
            self.assertIsNone(p.grad, name)
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.geometry_adapter.output.weight,
                                     before['geometry_adapter.output.weight']))
        model.load_state_dict(model.state_dict(), strict=True)

    def test_ignored_and_saturated_losses(self):
        for ignored in (True, False):
            model = build_v3(self.source).train()
            # Force extreme, finite anchor logits to cover probability saturation.
            torch.nn.init.zeros_(model.refiner.coarse_head.weight)
            torch.nn.init.constant_(model.refiner.coarse_head.bias, 80)
            image, sample = _sample(64)
            if ignored:
                sample.gt_sem_seg.data.fill_(255)
            losses = model.loss(_add_pseudo_geometry(image)[None], [sample])
            loss, _ = model.parse_losses(losses)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            if ignored:
                self.assertEqual(loss.item(), 0)
            for name, p in model.geometry_adapter.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_predict_slide_and_odd_sizes(self):
        model = build_v3(self.source).eval()
        model.test_cfg = dict(mode='slide', crop_size=(64, 64), stride=(48, 48))
        inputs = torch.rand(1, 5, 81, 97) * 255
        with torch.no_grad():
            predictions = model.predict(inputs)
            self.assertEqual(predictions[0].pred_sem_seg.data.shape, (1, 81, 97))
            logits = model.encode_decode(torch.rand(1, 5, 65, 67) * 255)
            self.assertEqual(logits.shape, (1, 2, 65, 67))
            self.assertTrue(torch.isfinite(logits).all())


if __name__ == '__main__':
    unittest.main()
