"""V6 worker-packed SDF targets must exactly match the original loss target."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from mmseg.registry import MODELS, TRANSFORMS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401
from wwtpseg.models.segmentors.rpgv_net_v2 import contour_target as original_contour_target
from wwtpseg.rpgv_v6_targets import contour_target_v6, get_contour_targets_v6


def results_for(mask):
    height, width = mask.shape
    image = np.zeros((height, width, 5), dtype=np.float32)
    image[..., 3] = np.linspace(0, 255, width)[None, :]
    image[..., 4] = 255
    return dict(img=image, gt_seg_map=mask.copy(), seg_fields=['gt_seg_map'],
                img_shape=mask.shape, ori_shape=mask.shape,
                scale_factor=(1.0, 1.0), reduce_zero_label=False)


def target_from(sample):
    label = sample.gt_sem_seg.data.unsqueeze(0)
    valid = label != 255
    return torch.where(valid, label, torch.zeros_like(label)), valid


class V6ContourTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        register_all_modules(init_default_scope=True)

    def setUp(self):
        self.packer = TRANSFORMS.build(dict(type='PackRPGVContourInputs', contour_truncation=5.0))

    def test_helper_is_exact_for_uniform_thin_ignored_and_odd_masks(self):
        for shape in ((64, 64), (63, 65)):
            height, width = shape
            thin = np.zeros(shape, dtype=np.uint8)
            thin[height // 2, 4:width - 4] = 1
            mixed = np.zeros(shape, dtype=np.uint8)
            mixed[8:height - 8, 9:width - 9] = 1
            mixed[height // 2 - 2:height // 2 + 2, :] = 255
            for mask in (np.zeros(shape, np.uint8), np.ones(shape, np.uint8),
                         np.full(shape, 255, np.uint8), thin, mixed):
                packed = self.packer(results_for(mask))
                sample = packed['data_samples']
                target, valid = target_from(sample)
                size = ((height + 1) // 2, (width + 1) // 2)
                expected = original_contour_target(target, valid, size, 5.0)
                actual = contour_target_v6(target, valid, size, 5.0)
                cached = get_contour_targets_v6(target, valid, size, 5.0, [sample])
                for first, second, third in zip(expected, actual, cached):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
                    torch.testing.assert_close(first, third, rtol=0, atol=0)

    def test_all_flips_cache_the_transformed_labels(self):
        mask = np.zeros((63, 65), dtype=np.uint8)
        mask[7:39, 11:28] = 1
        mask[:3] = 255
        for direction in ('horizontal', 'vertical', 'diagonal'):
            flip = TRANSFORMS.build(dict(type='RandomFlip', prob=1.0, direction=direction))
            sample = self.packer(flip(results_for(mask)))['data_samples']
            target, valid = target_from(sample)
            expected = original_contour_target(target, valid, (32, 33), 5.0)
            actual = get_contour_targets_v6(target, valid, (32, 33), 5.0, [sample])
            for first, second in zip(expected, actual):
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_matching_cache_is_used_without_edt(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:40, 20:50] = 1
        sample = self.packer(results_for(mask))['data_samples']
        target, valid = target_from(sample)
        with patch('wwtpseg.rpgv_v6_targets.contour_target_v6',
                   side_effect=AssertionError('cache unexpectedly bypassed')):
            sdf, known = get_contour_targets_v6(target, valid, (32, 32), 5.0, [sample])
        self.assertEqual(tuple(sdf.shape), (1, 1, 32, 32))
        self.assertEqual(known.dtype, torch.bool)

    def test_size_truncation_and_padding_mismatch_recompute(self):
        sample = self.packer(results_for(np.zeros((63, 63), np.uint8)))['data_samples']
        target, valid = target_from(sample)
        for size, truncation in (((31, 31), 5.0), ((32, 32), 3.0)):
            with patch('wwtpseg.rpgv_v6_targets.contour_target_v6',
                       wraps=contour_target_v6) as fallback:
                actual = get_contour_targets_v6(target, valid, size, truncation, [sample])
            self.assertEqual(fallback.call_count, 1)
            expected = original_contour_target(target, valid, size, truncation)
            for first, second in zip(actual, expected):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
        # The half-size is still 32x32, but one padded source row/column changes
        # occupancy and safe masking. Matching output dimensions are not enough.
        target = torch.nn.functional.pad(target, (0, 1, 0, 1))
        valid = torch.nn.functional.pad(valid, (0, 1, 0, 1), value=False)
        with patch('wwtpseg.rpgv_v6_targets.contour_target_v6',
                   wraps=contour_target_v6) as fallback:
            actual = get_contour_targets_v6(target, valid, (32, 32), 5.0, [sample])
        self.assertEqual(fallback.call_count, 1)
        for first, second in zip(actual, original_contour_target(target, valid, (32, 32), 5.0)):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_model_cache_and_fallback_losses_and_gradients_match(self):
        from tests.rpgv.test_rpgv_v6 import objective, tiny_model

        torch.manual_seed(7)
        mask = np.zeros((64, 64), np.uint8)
        mask[10:40, 20:50] = 1
        mask[:2] = 255
        packed = self.packer(results_for(mask))
        inputs = packed['inputs'].float().unsqueeze(0)
        sample = packed['data_samples']
        model = tiny_model().eval()
        with patch('wwtpseg.rpgv_v6_targets.contour_target_v6',
                   side_effect=AssertionError('model did not use packed SDF')):
            cached_losses = model.loss(inputs, [sample])
        objective(cached_losses).backward()
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
        plain = copy.deepcopy(sample)
        del plain.gt_contour_sdf
        del plain.gt_contour_valid
        model.zero_grad(set_to_none=True)
        fallback_losses = model.loss(inputs, [plain])
        objective(fallback_losses).backward()
        for name in cached_losses:
            torch.testing.assert_close(cached_losses[name], fallback_losses[name], rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0, msg=name)

    def test_real_preprocessor_padding_invalidates_cache(self):
        from tests.rpgv.test_rpgv_v6 import objective, tiny_model

        mask = np.zeros((63, 63), np.uint8)
        mask[12:39, 20:50] = 1
        packed = self.packer(results_for(mask))
        model = tiny_model().eval()
        processor = MODELS.build(dict(type='SegDataPreProcessor', size=(64, 64),
                                      pad_val=0, seg_pad_val=255, bgr_to_rgb=False))
        processed = processor(dict(inputs=[packed['inputs']],
                                   data_samples=[packed['data_samples']]), training=True)
        with patch('wwtpseg.rpgv_v6_targets.contour_target_v6',
                   wraps=contour_target_v6) as fallback:
            losses = model.loss(**processed)
        self.assertEqual(fallback.call_count, 1)
        self.assertTrue(torch.isfinite(objective(losses)))


if __name__ == '__main__':
    unittest.main()
