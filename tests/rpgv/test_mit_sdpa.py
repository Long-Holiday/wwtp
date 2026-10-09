"""SDPA-backed MiT registration, compatibility, gradients and CUDA v6 checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from mmseg.models.backbones import MixVisionTransformer
from torch.nn import functional as F

import wwtpseg  # noqa: F401
from wwtpseg.models.backbones.mit_sdpa import MixVisionTransformerSDPA, WeightFreeMultiheadAttention


BACKBONE = dict(
    in_channels=3, embed_dims=8, num_stages=4, num_layers=[1, 1, 1, 1],
    num_heads=[1, 2, 4, 8], patch_sizes=[7, 3, 3, 3], sr_ratios=[8, 4, 2, 1],
    out_indices=(0, 1, 2, 3), drop_path_rate=0.0)


class MiTSDPATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_state_keys_outputs_and_gradients_match_upstream(self):
        original = MixVisionTransformer(**BACKBONE)
        optimized = MixVisionTransformerSDPA(**BACKBONE)
        self.assertEqual(set(original.state_dict()), set(optimized.state_dict()))
        optimized.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        expected, actual = original(x), optimized(y)
        gradients = [torch.randn_like(feature) for feature in expected]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=5e-4, atol=2e-5)
        sum((a * g).sum() for a, g in zip(actual, gradients)).backward()
        sum((b * g).sum() for b, g in zip(expected, gradients)).backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-3, atol=5e-5)
        for (name, first), (other, second) in zip(original.named_parameters(), optimized.named_parameters()):
            self.assertEqual(name, other)
            self.assertIsNotNone(second.grad, name)
            torch.testing.assert_close(first.grad, second.grad, rtol=1e-3, atol=2e-4, msg=name)
        self.assertNotIsInstance(original.layers[0][1][0].attn.attn, WeightFreeMultiheadAttention)

    def test_sdpa_is_dispatched_and_explicit_weights_still_work(self):
        model = MixVisionTransformerSDPA(**BACKBONE).train()
        with patch('torch.nn.functional.scaled_dot_product_attention',
                   wraps=F.scaled_dot_product_attention) as attention:
            model(torch.randn(2, 3, 64, 64))
        self.assertGreaterEqual(attention.call_count, 4)
        leaf = model.layers[0][1][0].attn.attn
        query = torch.randn(5, 2, 8)
        _, weights = leaf(query, query, query)
        self.assertIsNone(weights)
        _, weights = leaf(query, query, query, need_weights=True)
        self.assertIsNotNone(weights)

    def test_local_pretrained_initialization_keeps_upstream_keys(self):
        reference = MixVisionTransformer(**BACKBONE)
        with tempfile.TemporaryDirectory(prefix='v6-mit-') as directory:
            checkpoint = Path(directory) / 'mit.pth'
            torch.save(dict(state_dict=reference.state_dict()), checkpoint)
            model = MixVisionTransformerSDPA(
                **BACKBONE, init_cfg=dict(type='Pretrained', checkpoint=str(checkpoint)))
            model.init_weights()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    def test_activation_checkpointing_has_finite_gradients_and_updates(self):
        model = MixVisionTransformerSDPA(**BACKBONE, with_cp=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = next(model.parameters()).detach().clone()
        features = model(torch.randn(2, 3, 64, 64))
        sum((value * torch.randn_like(value)).mean() for value in features).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertGreater((next(model.parameters()).detach() - before).abs().max().item(), 0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class V6CUDAAMPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fp16_training_missing_geometry_ignore_and_dropout(self):
        from tests.rpgv.test_rpgv_v6 import batch, objective, tiny_model

        for case in ('normal', 'mixed_missing', 'zero_quality', 'all_ignore', 'dropout'):
            torch.manual_seed(42)
            model = tiny_model(geometry_dropout_prob=1 if case == 'dropout' else 0).cuda().train()
            inputs, samples = batch(label=255 if case == 'all_ignore' else None)
            inputs = inputs.cuda()
            if case == 'mixed_missing':
                inputs[0, 4] = 0
            elif case == 'zero_quality':
                inputs[:, 4] = 0
            inputs.requires_grad_(True)
            with torch.autocast('cuda', dtype=torch.float16):
                losses = model.loss(inputs, samples)
                total = objective(losses)
            self.assertTrue(torch.isfinite(total), case)
            total.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, (case, name))
                self.assertTrue(torch.isfinite(parameter.grad).all(), (case, name))
            if case in {'mixed_missing', 'zero_quality', 'all_ignore', 'dropout'}:
                self.assertEqual(inputs.grad[0, 3].abs().sum().item(), 0, case)

    def test_trained_fp16_zero_quality_exact_rgb_fallback(self):
        from tests.rpgv.test_rpgv_v6 import batch, objective, tiny_model

        model = tiny_model().cuda().train()
        inputs, samples = batch(batch_size=1)
        inputs = inputs.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        objective(model.loss(inputs, samples)).backward()
        optimizer.step()
        model.eval()
        inputs[:, 4] = 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            reference = model._run_rgb_branch(inputs)['seg_logits']
            actual = model._run_network(inputs)['seg_logits']
        self.assertTrue(torch.equal(actual, reference))


if __name__ == '__main__':
    unittest.main()
