"""CPU regression tests for RPGV v6 (run with python -m unittest)."""

from __future__ import annotations

import copy
import unittest

import torch
from mmengine.config import Config
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
from mmseg.structures import SegDataSample
from torch import nn
from torch.nn import functional as F

import wwtpseg  # noqa: F401
from wwtpseg.models.utils.rpgv_v6_modules import (
    DeformableDepthRectifierV6,
    GatedGeometryFusionV6,
    LowFrequencyCrossAttentionV6,
    RegionContrastiveLossV6,
    SmoothFrequencySplitV6,
    SpatialGeometryClusteringV6,
)


def tiny_model(**overrides):
    config = dict(
        type='RPGVNetV6',
        rgb_encoder=dict(
            type='MixVisionTransformer', in_channels=3, embed_dims=8,
            num_stages=4, num_layers=[1, 1, 1, 1], num_heads=[1, 2, 4, 8],
            patch_sizes=[7, 3, 3, 3], sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3), drop_path_rate=0.0),
        rgb_channels=(8, 16, 32, 64), geometry_channels=(8, 16),
        decoder_channels=16, detail_channels=8, rectifier_channels=8,
        geometry_dropout_prob=0.0,
        cluster_cfg=dict(grid_size=2, max_cluster_size=16, min_region_mass=1),
        attention_cfg=dict(query_size=16, kv_size=4),
    )
    config.update(overrides)
    return MODELS.build(config)


def batch(size=(64, 64), batch_size=2, label=None):
    height, width = size
    inputs = torch.rand(batch_size, 5, height, width) * 255
    inputs[:, 4] = 255
    samples = []
    for _ in range(batch_size):
        mask = (torch.randint(0, 2, (1, height, width)) if label is None
                else torch.full((1, height, width), label, dtype=torch.long))
        sample = SegDataSample(gt_sem_seg=PixelData(data=mask))
        sample.set_metainfo(dict(
            ori_shape=size, img_shape=size, pad_shape=size,
            padding_size=[0, 0, 0, 0], scale_factor=(1.0, 1.0)))
        samples.append(sample)
    return inputs, samples


def objective(losses):
    return sum(value for name, value in losses.items() if 'loss' in name)


class V6ModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_activation_backward_handles_mixed_memory_layouts(self):
        activation = SpatialGeometryClusteringV6(8).context[-1]
        value = torch.randn(2, 8, 16, 16, requires_grad=True)
        gradient = torch.randn_like(value).contiguous(memory_format=torch.channels_last)
        gradient[0] = 0
        actual = torch.autograd.grad(activation(value), value, gradient)[0]
        expected = torch.autograd.grad(activation(value), value, gradient.contiguous())[0]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual[0].abs().sum().item(), 0)

    def test_frequency_complementary_odd_sizes_and_gradient(self):
        depth = torch.rand(2, 1, 17, 19, requires_grad=True)
        low, high = SmoothFrequencySplitV6()(depth)
        torch.testing.assert_close(low + high, depth)
        (low.square().mean() + high.square().mean()).backward()
        self.assertTrue(torch.isfinite(depth.grad).all())
        self.assertGreater(depth.grad.abs().sum().item(), 0)

    def test_frequency_constant_and_half_precision(self):
        split = SmoothFrequencySplitV6()
        low, high = split(torch.ones(1, 1, 15, 17))
        torch.testing.assert_close(low, torch.ones_like(low))
        self.assertLess(high.abs().max().item(), 1e-6)
        depth = torch.rand(1, 1, 15, 17, dtype=torch.float16, requires_grad=True)
        low, high = split(depth)
        self.assertEqual(low.dtype, torch.float16)
        (low.float().square().mean() + high.float().square().mean()).backward()
        self.assertTrue(torch.isfinite(depth.grad).all())

    def test_rectifier_bounds_and_mask_before_spatial_mixing(self):
        module = DeformableDepthRectifierV6(channels=8)
        depth = torch.rand(1, 1, 33, 35)
        quality = torch.ones_like(depth)
        quality[..., :17] = 0
        corrupted = torch.where(quality == 0, torch.full_like(depth, 1e6), depth)
        first = module(depth, quality, (9, 9))
        second = module(corrupted, quality, (9, 9))
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        self.assertLessEqual(first['offsets'].abs().max().item(), module.max_offset)
        self.assertLessEqual(first['correction'].abs().max().item(), module.max_correction)
        first['corrected_depth'].square().mean().backward()
        self.assertGreater(module.offset_mask.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(module.offset_mask.weight.grad).all())

    def test_rectifier_preserves_low_confidence_attenuation(self):
        module = DeformableDepthRectifierV6(channels=8)
        depth = torch.rand(1, 1, 33, 35)
        quality = torch.full_like(depth, 1e-4)
        outputs = module(depth, quality, (9, 9))
        self.assertLessEqual(outputs['corrected_depth'].abs().max().item(), 1.01e-4)
        _, high = SmoothFrequencySplitV6()(outputs['corrected_depth'])
        self.assertLess(high.abs().max().item(), 1e-4)

    def test_contrastive_pooling_cannot_move_assignments(self):
        module = SpatialGeometryClusteringV6(8, grid_size=2, min_region_mass=1)
        geometry = torch.randn(1, 8, 9, 11, requires_grad=True)
        rgb = torch.randn_like(geometry, requires_grad=True)
        quality = torch.ones(1, 1, 9, 11)
        outputs = module(geometry, rgb, quality, quality)
        gradient = torch.autograd.grad(outputs['rgb_prototypes'].sum(), geometry,
                                       allow_unused=True, retain_graph=True)[0]
        self.assertIsNone(gradient)
        outputs['geometry_prototypes'].square().mean().backward()
        self.assertGreater(geometry.grad.abs().sum().item(), 0)

    def test_same_region_assignments_pool_corresponding_rgb(self):
        module = SpatialGeometryClusteringV6(8, grid_size=2, min_region_mass=1)
        features = torch.randn(2, 8, 9, 11, requires_grad=True)
        quality = torch.ones(2, 1, 9, 11)
        outputs = module(features, features, quality, quality)
        torch.testing.assert_close(outputs['geometry_prototypes'], outputs['rgb_prototypes'])
        self.assertTrue(outputs['valid_regions'].all())
        self.assertEqual(outputs['detail'].shape, features.shape)
        outputs['detail'].square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_low_reliability_flat_and_empty_regions_are_filtered(self):
        module = SpatialGeometryClusteringV6(8, grid_size=2)
        features = torch.randn(2, 8, 9, 11)
        quality = torch.ones(2, 1, 9, 11)
        for reliability, energy in ((quality * 0.1, quality), (quality, quality * 0),
                                    (quality * 0, quality)):
            outputs = module(features, features, reliability, energy)
            self.assertFalse(outputs['valid_regions'].any())
            self.assertEqual(outputs['detail'].abs().max().item(), 0)
            self.assertTrue(torch.isfinite(outputs['geometry_prototypes']).all())

    def test_empty_regions_stay_invalid_with_zero_thresholds(self):
        module = SpatialGeometryClusteringV6(
            8, grid_size=2, min_reliability=0, min_region_mass=0, min_region_energy=0)
        features = torch.randn(1, 8, 9, 11)
        quality = torch.zeros(1, 1, 9, 11)
        outputs = module(features, features, quality, quality)
        self.assertFalse(outputs['valid_regions'].any())

    def test_bidirectional_infonce_matches_reference(self):
        module = RegionContrastiveLossV6(4, 4, temperature=0.2)
        module.geometry_projection = nn.Identity()
        module.rgb_projection = nn.Identity()
        geometry = torch.randn(2, 3, 4, requires_grad=True)
        rgb = torch.randn(2, 3, 4, requires_grad=True)
        valid = torch.tensor([[True, False, True], [True, True, False]])
        geo = F.normalize(geometry[valid], dim=-1)
        appearance = F.normalize(rgb[valid], dim=-1)
        logits = geo @ appearance.T / 0.2
        labels = torch.arange(4)
        expected = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        actual = module(geometry, rgb, valid)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        for grad in (geometry.grad, rgb.grad, module.logit_scale.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
        self.assertEqual(geometry.grad[~valid].abs().sum().item(), 0)
        self.assertEqual(rgb.grad[~valid].abs().sum().item(), 0)

    def test_positive_correspondence_matters(self):
        module = RegionContrastiveLossV6(4, 4)
        module.geometry_projection = nn.Identity()
        module.rgb_projection = nn.Identity()
        features = torch.eye(4).unsqueeze(0)
        valid = torch.ones(1, 4, dtype=torch.bool)
        paired = module(features, features, valid)
        shuffled = module(features, features.roll(1, dims=1), valid)
        self.assertLess(paired.item(), shuffled.item())

    def test_insufficient_pairs_return_attached_zero(self):
        for count in (0, 1):
            module = RegionContrastiveLossV6(8)
            geometry = torch.randn(1, 4, 8, requires_grad=True)
            rgb = torch.randn(1, 4, 8, requires_grad=True)
            valid = torch.arange(4).unsqueeze(0) < count
            loss = module(geometry, rgb, valid)
            self.assertEqual(loss.item(), 0)
            loss.backward()
            for parameter in list(module.parameters()) + [geometry, rgb]:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_contrastive_ignores_invalid_regions(self):
        module = RegionContrastiveLossV6(8)
        geometry, rgb = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        valid = torch.tensor([[True, False, True, False], [True, True, False, False]])
        original = module(geometry, rgb, valid)
        geometry[~valid] = 1e4
        rgb[~valid] = -1e4
        torch.testing.assert_close(module(geometry, rgb, valid), original)

    def test_cross_attention_empty_keys_is_finite_zero(self):
        module = LowFrequencyCrossAttentionV6(8, query_size=8, kv_size=4)
        rgb = torch.randn(1, 8, 13, 15, requires_grad=True)
        low = torch.randn_like(rgb, requires_grad=True)
        output = module(rgb, low, torch.zeros(1, 1, 13, 15))
        self.assertEqual(output.abs().max().item(), 0)
        output.sum().backward()
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_cross_attention_has_gradients_in_both_modalities(self):
        module = LowFrequencyCrossAttentionV6(8, query_size=8, kv_size=4)
        rgb = torch.randn(1, 8, 13, 15, requires_grad=True)
        low = torch.randn_like(rgb, requires_grad=True)
        output = module(rgb, low, torch.ones(1, 1, 13, 15))
        self.assertEqual(output.shape, rgb.shape)
        self.assertTrue(output.is_contiguous())
        output.square().mean().backward()
        for parameter in (rgb, low, module.query.weight, module.key.weight, module.value.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_gate_zero_reliability_is_exact_identity(self):
        module = GatedGeometryFusionV6(8)
        rgb = torch.randn(2, 8, 7, 9)
        fused, gates = module(rgb, torch.randn_like(rgb), torch.randn_like(rgb),
                              torch.zeros(2, 1, 7, 9))
        self.assertTrue(torch.equal(fused, rgb))
        self.assertTrue(((gates > 0) & (gates < 1)).all())

    def test_configuration_validation(self):
        for cutoff in (0, -1, 0.6, float('nan')):
            with self.assertRaises(ValueError):
                SmoothFrequencySplitV6(cutoff)
        with self.assertRaises(ValueError):
            LowFrequencyCrossAttentionV6(7, num_heads=4)
        with self.assertRaises(ValueError):
            SpatialGeometryClusteringV6(8, min_reliability=2)
        with self.assertRaises(ValueError):
            RegionContrastiveLossV6(8, temperature=0)


class V6ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        register_all_modules(init_default_scope=True)

    def test_production_and_ablation_configs(self):
        for variant in ('rpgv_v6', 'no_contrastive', 'rgb_only'):
            config = Config.fromfile(f'configs/rpgv/v6/{variant}.py')
            self.assertEqual(config.model.type, 'RPGVNetV6')
            self.assertEqual(tuple(config.model.data_preprocessor.size), (1024, 1024))
            self.assertEqual(config.train_dataloader.batch_size, 2)
            self.assertEqual(config.optim_wrapper.accumulative_counts, 4)
            self.assertEqual(config.train_cfg.max_iters, 20000)
            self.assertIsNone(config.load_from)
            self.assertFalse(config.resume)
            self.assertNotIn('required_previous_stage', config)
            self.assertEqual(config.optim_wrapper.paramwise_cfg.custom_keys.logit_scale.decay_mult, 0)
            for split in ('train', 'val', 'test'):
                steps = config[f'{split}_dataloader'].dataset.pipeline
                names = [step['type'] for step in steps]
                self.assertIn('LoadPseudoGeometry', names)
                self.assertIn('RequireImageShape', names)
                self.assertNotIn('RandomCrop', names)
                self.assertNotIn('Resize', names)
            config.model.rgb_encoder.init_cfg = None
            model = MODELS.build(config.model)
            self.assertEqual(model.use_geometry, variant != 'rgb_only')
            self.assertEqual(hasattr(model, 'region_contrastive'), variant == 'rpgv_v6')


class V6SegmentorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def test_loss_backward_step_and_strict_reload(self):
        model = tiny_model()
        inputs, samples = batch()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(inputs, samples)
            self.assertTrue({'loss_final', 'loss_coarse', 'loss_sdf', 'loss_boundary_band',
                             'loss_region_contrastive', 'loss_depth_rectification'} <= losses.keys())
            self.assertGreater(losses['region_valid_count'].item(), 1)
            total = objective(losses)
            self.assertTrue(torch.isfinite(total))
            total.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer.step()
        replica = tiny_model()
        replica.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        model.eval()
        replica.eval()
        with torch.no_grad():
            torch.testing.assert_close(model._run_network(inputs)['seg_logits'],
                                       replica._run_network(inputs)['seg_logits'])

    def test_segmentation_trains_each_geometry_path(self):
        model = tiny_model()
        inputs, _ = batch()
        model._run_network(inputs)['final_logits'].square().mean().backward()
        parameters = dict(model.named_parameters())
        for name in ('depth_rectifier.offset_mask.weight', 'depth_rectifier.deform.weight',
                     'depth_rectifier.residual.weight', 'geometry_stages.0.high_encoder.0.weight',
                     'geometry_stages.0.low_attention.query.weight',
                     'geometry_stages.0.low_attention.key.weight',
                     'geometry_stages.0.low_attention.value.weight',
                     'geometry_stages.0.fusion.gates.2.weight'):
            self.assertGreater(parameters[name].grad.abs().sum().item(), 0, name)
            self.assertTrue(torch.isfinite(parameters[name].grad).all(), name)

    def test_zero_confidence_and_rgb_input_exact_fallback(self):
        model = tiny_model().eval()
        inputs, _ = batch(batch_size=1)
        inputs[:, 4] = 0
        with torch.no_grad():
            expected = model._run_rgb_branch(inputs)['seg_logits']
            for value in (inputs, inputs[:, :3]):
                actual = model._run_network(value)['seg_logits']
                self.assertTrue(torch.equal(actual, expected))

    def test_untrusted_depth_never_leaks_to_valid_neighbours(self):
        model = tiny_model().eval()
        inputs, _ = batch(batch_size=1)
        inputs[:, 4, :, :32] = 0
        changed = inputs.clone()
        changed[:, 3, :, :32] = 255 - changed[:, 3, :, :32]
        with torch.no_grad():
            self.assertTrue(torch.equal(model._run_network(inputs)['seg_logits'],
                                        model._run_network(changed)['seg_logits']))

    def test_all_ignore_has_zero_attached_objectives(self):
        model = tiny_model()
        inputs, samples = batch(label=255)
        losses = model.loss(inputs, samples)
        self.assertEqual(objective(losses).item(), 0)
        self.assertEqual(losses['region_valid_count'].item(), 0)
        objective(losses).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_geometry_dropout_keeps_gradients_and_zero_alignment(self):
        model = tiny_model(geometry_dropout_prob=1.0).train()
        inputs, samples = batch()
        losses = model.loss(inputs, samples)
        self.assertEqual(losses['loss_region_contrastive'].item(), 0)
        self.assertEqual(losses['loss_depth_rectification'].item(), 0)
        objective(losses).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_mixed_missing_geometry_batch_has_finite_isolated_gradients(self):
        for seed in (10, 33, 53, 87):
            torch.manual_seed(seed)
            model = tiny_model()
            inputs, samples = batch()
            inputs[0, 4] = 0
            inputs.requires_grad_(True)
            losses = model.loss(inputs, samples)
            self.assertGreater(losses['region_valid_count'].item(), 1)
            objective(losses).backward()
            self.assertEqual(inputs.grad[0, 3].abs().sum().item(), 0)
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), f'{seed}: {name}')

    def test_empty_foreground_loss_is_finite(self):
        inputs, samples = batch(label=0)
        losses = tiny_model().loss(inputs, samples)
        self.assertTrue(torch.isfinite(objective(losses)))

    def test_loss_in_eval_mode_and_odd_prediction(self):
        model = tiny_model().eval()
        inputs, samples = batch((65, 67), batch_size=1)
        with torch.no_grad():
            losses = model.loss(inputs, samples)
            self.assertTrue(torch.isfinite(losses['loss_region_contrastive']))
            predictions = model.predict(inputs, samples)
        self.assertEqual(tuple(predictions[0].pred_sem_seg.data.shape), (1, 65, 67))

    def test_slide_inference(self):
        model = tiny_model(test_cfg=dict(mode='slide', crop_size=(64, 64),
                                         stride=(48, 48))).eval()
        inputs, samples = batch((80, 83), batch_size=1)
        with torch.no_grad():
            predictions = model.predict(inputs, samples)
        self.assertEqual(tuple(predictions[0].pred_sem_seg.data.shape), (1, 80, 83))
        self.assertTrue(torch.isfinite(predictions[0].seg_logits.data).all())

    def test_ablations_do_not_leave_unused_trainable_branches(self):
        inputs, samples = batch()
        for overrides in (dict(use_geometry=False), dict(contrastive_loss_weight=0.0)):
            model = tiny_model(**overrides)
            self.assertFalse(hasattr(model, 'region_contrastive'))
            losses = model.loss(inputs, samples)
            self.assertNotIn('loss_region_contrastive', losses)
            objective(losses).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_cpu_bfloat16_autocast_training(self):
        model = tiny_model()
        inputs, samples = batch()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            losses = model.loss(inputs, samples)
            total = objective(losses)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == '__main__':
    unittest.main()
