"""CPU regression tests for RPGV v6 (run with python -m unittest)."""

from __future__ import annotations

import copy
import unittest

from unittest.mock import patch

import torch
from mmengine.config import Config
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
from mmseg.structures import SegDataSample
from torch import nn
from torch.nn import functional as F

import wwtpseg  # noqa: F401
from wwtpseg.models.segmentors.multiclass import RPGVRemoteSensingV6
from wwtpseg.models.utils.rpgv_v6_decoder import ResidualContourHeadV6, SpatialGeometryDecoderV6
from wwtpseg.models.utils.rpgv_v6_modules import (
    DeformableDepthRectifierV6,
    FrequencyGeometryStageV6,
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
            type='MixVisionTransformerSDPA', in_channels=3, embed_dims=8,
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
        self.assertLessEqual(outputs['trusted_depth'].abs().max().item(), 1.01e-4)
        torch.testing.assert_close(outputs['trusted_depth'],
                                   outputs['corrected_depth'] * outputs['reliability'])
        low, high = SmoothFrequencySplitV6()(outputs['corrected_depth'], outputs['reliability'])
        torch.testing.assert_close(low + high, outputs['trusted_depth'])
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
            self.assertEqual(outputs['detail'].shape, features.shape)
            self.assertTrue(torch.isfinite(outputs['detail']).all())
            self.assertTrue(torch.isfinite(outputs['geometry_prototypes']).all())

    def test_empty_regions_stay_invalid_with_zero_thresholds(self):
        module = SpatialGeometryClusteringV6(
            8, grid_size=2, min_reliability=0, min_region_mass=0, min_region_energy=0)
        features = torch.randn(1, 8, 9, 11)
        quality = torch.zeros(1, 1, 9, 11)
        outputs = module(features, features, quality, quality)
        self.assertFalse(outputs['valid_regions'].any())

    def test_siglip_matches_reference(self):
        module = RegionContrastiveLossV6(4, 4, temperature=0.2, logit_bias=-1.5)
        module.geometry_projection = nn.Identity()
        module.rgb_projection = nn.Identity()
        geometry = torch.randn(2, 3, 4, requires_grad=True)
        rgb = torch.randn(2, 3, 4, requires_grad=True)
        valid = torch.tensor([[True, False, True], [True, True, False]])
        geo = F.normalize(geometry[valid], dim=-1)
        appearance = F.normalize(rgb[valid], dim=-1)
        logits = geo @ appearance.T / 0.2 - 1.5
        labels = torch.eye(4)
        expected = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum') / 4
        actual = module(geometry, rgb, valid)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        for grad in (geometry.grad, rgb.grad, module.logit_scale.grad, module.logit_bias.grad):
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

    def test_no_pairs_return_attached_zero(self):
        module = RegionContrastiveLossV6(8)
        geometry = torch.randn(1, 4, 8, requires_grad=True)
        rgb = torch.randn(1, 4, 8, requires_grad=True)
        valid = torch.zeros(1, 4, dtype=torch.bool)
        loss = module(geometry, rgb, valid)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        for parameter in list(module.parameters()) + [geometry, rgb]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertEqual(parameter.grad.abs().sum().item(), 0)

    def test_single_pair_has_positive_loss_and_gradients(self):
        torch.manual_seed(13)
        module = RegionContrastiveLossV6(8)
        geometry = torch.randn(1, 4, 8, requires_grad=True)
        rgb = torch.randn(1, 4, 8, requires_grad=True)
        valid = torch.tensor([[False, False, True, False]])
        geo = F.normalize(module.geometry_projection(geometry[valid]), dim=-1)
        appearance = F.normalize(module.rgb_projection(rgb[valid]), dim=-1)
        logit = (geo * appearance).sum() * module.logit_scale.exp() + module.logit_bias
        expected = F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))
        loss = module(geometry, rgb, valid)
        torch.testing.assert_close(loss, expected)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        for parameter in list(module.parameters()) + [geometry, rgb]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertEqual(geometry.grad[~valid].abs().sum().item(), 0)
        self.assertEqual(rgb.grad[~valid].abs().sum().item(), 0)

    def test_siglip_initialization_and_finite_bias(self):
        module = RegionContrastiveLossV6(8)
        torch.testing.assert_close(module.logit_scale.exp(), torch.tensor(10.0))
        self.assertEqual(module.logit_bias.item(), -10)
        for bias in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                RegionContrastiveLossV6(8, logit_bias=bias)

    def test_siglip_large_bias_is_stable(self):
        for bias in (-1000.0, 1000.0):
            module = RegionContrastiveLossV6(8, logit_bias=bias)
            geometry = torch.randn(1, 3, 8, requires_grad=True)
            rgb = torch.randn(1, 3, 8, requires_grad=True)
            loss = module(geometry, rgb, torch.ones(1, 3, dtype=torch.bool))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for parameter in list(module.parameters()) + [geometry, rgb]:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_siglip_cpu_autocast_matches_fp32(self):
        module = RegionContrastiveLossV6(8)
        geometry, rgb = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        valid = torch.tensor([[True, False, True, False], [True, True, False, False]])
        reference = module(geometry, rgb, valid)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = module(geometry, rgb, valid)
        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, reference)

    def test_siglip_scale_preserves_existing_bounds(self):
        module = RegionContrastiveLossV6(4, 4, logit_bias=0.0)
        module.geometry_projection = nn.Identity()
        module.rgb_projection = nn.Identity()
        features = torch.eye(4).unsqueeze(0)
        valid = torch.ones(1, 4, dtype=torch.bool)
        for raw_scale, expected_scale in ((-2.0, 1.0), (10.0, 100.0)):
            with torch.no_grad():
                module.logit_scale.fill_(raw_scale)
            logits = torch.eye(4) * expected_scale
            expected = F.binary_cross_entropy_with_logits(
                logits, torch.eye(4), reduction='sum') / 4
            torch.testing.assert_close(module(features, features, valid), expected)

    def test_contrastive_ignores_invalid_regions(self):
        module = RegionContrastiveLossV6(8)
        geometry, rgb = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        valid = torch.tensor([[True, False, True, False], [True, True, False, False]])
        original = module(geometry, rgb, valid)
        geometry[~valid] = 1e4
        rgb[~valid] = -1e4
        torch.testing.assert_close(module(geometry, rgb, valid), original)

    def test_invalid_nonfinite_contrastive_inputs_are_not_used(self):
        module = RegionContrastiveLossV6(8)
        valid = torch.tensor([[True, False, True, False]])
        geometry = torch.randn(1, 4, 8)
        rgb = torch.randn(1, 4, 8)
        reference = module(geometry, rgb, valid)
        geometry[~valid] = float('nan')
        rgb[~valid] = float('inf')
        geometry.requires_grad_(True)
        rgb.requires_grad_(True)
        loss = module(geometry, rgb, valid)
        torch.testing.assert_close(loss, reference)
        loss.backward()
        for parameter in list(module.parameters()) + [geometry, rgb]:
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(geometry.grad[~valid].abs().sum().item(), 0)
        self.assertEqual(rgb.grad[~valid].abs().sum().item(), 0)

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
        delta, gates = module(rgb, torch.randn_like(rgb), torch.randn_like(rgb),
                              torch.zeros(2, 1, 7, 9))
        self.assertEqual(delta.abs().max().item(), 0)
        self.assertTrue(torch.equal(rgb + delta, rgb))
        self.assertTrue(((gates > 0) & (gates < 1)).all())

    def test_frequency_holes_soft_confidence_and_small_support(self):
        split = SmoothFrequencySplitV6()
        for size in ((17, 19), (1, 7), (5, 1), (1, 1)):
            depth = torch.full((1, 1, *size), 0.5, requires_grad=True)
            quality = torch.ones_like(depth)
            if size[0] > 1 and size[1] > 1:
                quality[..., 4:11, 5:13] = 0
                quality[..., :3, :4] = 0.4
            for scale in (1.0, 1e-9):
                q = quality * scale
                low, high = split(depth, q)
                torch.testing.assert_close(low + high, depth * q)
                self.assertLess(high.abs().max().item(), 2e-7 * scale)
                corrupted = torch.where(q > 0, depth, torch.full_like(depth, 1e6))
                other = split(corrupted, q)
                torch.testing.assert_close(low, other[0], rtol=0, atol=0)
                torch.testing.assert_close(high, other[1], rtol=0, atol=0)
            (low.square().sum() + high.square().sum()).backward()
            self.assertTrue(torch.isfinite(depth.grad).all())

    def test_rectifier_derivatives_do_not_see_missing_neighbour_edges(self):
        module = DeformableDepthRectifierV6(channels=8)
        depth = torch.full((1, 1, 17, 19), 0.5)
        quality = torch.ones_like(depth)
        quality[..., 5:12, 6:14] = 0
        captured = []
        handle = module.encoder.register_forward_pre_hook(lambda m, args: captured.append(args[0]))
        module(depth, quality, depth.shape[-2:])
        handle.remove()
        self.assertEqual(captured[0][:, 2:].abs().max().item(), 0)

    def test_gain_and_gate_layers_learn_on_first_step(self):
        module = GatedGeometryFusionV6(8)
        rgb = torch.randn(2, 8, 7, 9)
        delta, gates = module(rgb, torch.randn_like(rgb), torch.randn_like(rgb),
                              torch.full((2, 1, 7, 9), 0.34))
        self.assertAlmostEqual(gates.mean().item(), 0.5, places=2)
        delta.square().mean().backward()
        for parameter in module.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_contour_residual_can_correct_confident_wrong_base(self):
        head = ResidualContourHeadV6(16, 8)
        with torch.no_grad():
            head.coarse_head.weight.zero_()
            head.coarse_head.bias.fill_(8)
            head.correction_head.weight.zero_()
            head.correction_head.bias.fill_(-12)
        rgb, feature = torch.randn(1, 3, 32, 34), torch.randn(1, 16, 8, 9)
        output = head(rgb, feature)
        self.assertTrue((output['coarse_logits'] > 0).all())
        self.assertTrue((output['final_logits'] < 0).all())
        self.assertGreater(output['contour_correction'].abs().min().item(), 2)
        with torch.no_grad():
            head.sdf_head.bias.add_(50)
        torch.testing.assert_close(head(rgb, feature)['final_logits'], output['final_logits'])

    def test_contour_shared_trunk_gets_immediate_segmentation_gradient(self):
        head = ResidualContourHeadV6(16, 8)
        output = head(torch.randn(2, 3, 32, 34), torch.randn(2, 16, 8, 9))
        output['final_logits'].square().mean().backward()
        for parameter in (head.rgb_stem[0][0].weight, head.contour[0][0].weight,
                          head.correction_head.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertIsNone(head.sdf_head.weight.grad)

    def test_decoder_callback_uses_pure_lateral_features(self):
        decoder = SpatialGeometryDecoderV6((8, 16, 32, 64), 16, use_context=False)
        features = (torch.randn(1, 8, 16, 16), torch.randn(1, 16, 8, 8),
                    torch.randn(1, 32, 4, 4), torch.randn(1, 64, 2, 2))
        references = [decoder.projections[i](features[i]) for i in range(2)]
        seen = []

        def fusion(index, value, reference):
            seen.append(index)
            torch.testing.assert_close(reference, references[index])
            return value + 10

        output = decoder(features, fusion_callback=fusion)
        self.assertEqual(seen, [1, 0])
        self.assertEqual(tuple(output.shape), (1, 16, 16, 16))

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
        for variant in ('rpgv_v6', 'no_contrastive', 'rgb_only', 'encoder_fusion',
                        'legacy_contour', 'cluster32'):
            config = Config.fromfile(f'configs/rpgv/v6/{variant}.py')
            self.assertEqual(config.model.type, 'RPGVNetV6')
            self.assertEqual(tuple(config.model.data_preprocessor.size), (1024, 1024))
            self.assertEqual(config.train_dataloader.batch_size, 2)
            self.assertEqual(config.optim_wrapper.accumulative_counts, 4)
            self.assertEqual(config.train_cfg.max_iters, 20000)
            self.assertIsNone(config.load_from)
            self.assertFalse(config.resume)
            expected_dir = ('work_dirs/rpgv_v6_siglip_1m' if variant == 'rpgv_v6'
                            else f'work_dirs/rpgv_v6_siglip_{variant}_1m')
            self.assertEqual(config.work_dir, expected_dir)
            self.assertNotIn('required_previous_stage', config)
            self.assertEqual(config.optim_wrapper.type, 'OptimWrapper')
            self.assertEqual(config.optim_wrapper.paramwise_cfg, dict(norm_decay_mult=0.0))
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
            self.assertEqual(hasattr(model, 'region_contrastive'),
                             variant not in {'rgb_only', 'no_contrastive'})
            self.assertEqual(config.model.contrastive_temperature, 0.1)
            self.assertEqual(config.model.contrastive_logit_bias, -10)
            if hasattr(model, 'region_contrastive'):
                for head in model.region_contrastive:
                    torch.testing.assert_close(head.logit_scale.exp(), torch.tensor(10.0))
                    self.assertEqual(head.logit_bias.item(), -10)
            self.assertEqual(config.model.geometry_dropout_prob, 0)
            self.assertEqual(config.model.rgb_encoder.type, 'MixVisionTransformerSDPA')
            self.assertTrue(config.model.rgb_encoder.with_cp)
            self.assertEqual(config.train_dataloader.dataset.pipeline[-1]['type'],
                             'PackRPGVContourInputs')


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
        state = copy.deepcopy(model.state_dict())
        old_state = {name: value for name, value in state.items()
                     if not name.endswith('.logit_bias')}
        with self.assertRaisesRegex(RuntimeError, 'logit_bias'):
            replica.load_state_dict(old_state, strict=True)
        replica.load_state_dict(state, strict=True)
        for head, restored in zip(model.region_contrastive, replica.region_contrastive):
            torch.testing.assert_close(restored.logit_bias, head.logit_bias)
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

    def test_low_quality_still_trains_high_frequency_segmentation(self):
        for quality in (0.2, 0.34):
            model = tiny_model()
            inputs, _ = batch()
            inputs[:, 4] = quality * 255
            output = model._run_network(inputs)
            self.assertFalse(any(r['valid_regions'].any() for r in output['geometry_v6']['regions']))
            output['final_logits'].square().mean().backward()
            for stage in model.geometry_stages:
                gradient = stage.high_encoder[0].weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)

    def test_prototypes_keep_modalities_independent(self):
        model = tiny_model()
        inputs, _ = batch()
        inputs.requires_grad_(True)
        output = model._run_network(inputs)
        regions = output['geometry_v6']['regions']
        geometry = sum(r['geometry_prototypes'].square().sum() for r in regions)
        appearance = sum(r['rgb_prototypes'].square().sum() for r in regions)
        geo_gradient = torch.autograd.grad(geometry, inputs, retain_graph=True)[0]
        rgb_gradient = torch.autograd.grad(appearance, inputs)[0]
        self.assertEqual(geo_gradient[:, :3].abs().sum().item(), 0)
        self.assertGreater(geo_gradient[:, 3:4].abs().sum().item(), 0)
        self.assertEqual(rgb_gradient[:, 3:4].abs().sum().item(), 0)
        self.assertGreater(rgb_gradient[:, :3].abs().sum().item(), 0)

    def test_disabled_legacy_pair_losses_are_not_computed(self):
        inputs, samples = batch()
        with patch('wwtpseg.models.segmentors.rpgv_net_v6.region_and_boundary_loss',
                   side_effect=AssertionError('disabled loss evaluated')):
            losses = tiny_model().loss(inputs, samples)
        self.assertTrue(torch.isfinite(objective(losses)))

    def test_architecture_and_head_ablations_have_no_unused_parameters(self):
        inputs, samples = batch()
        for overrides in (dict(fusion_location='encoder'), dict(contour_mode='legacy'),
                          dict(sdf_loss_weight=0), dict(use_contour=False),
                          dict(use_contour=False, sdf_loss_weight=0)):
            model = tiny_model(**overrides)
            losses = model.loss(inputs, samples)
            objective(losses).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, (overrides, name))
                self.assertTrue(torch.isfinite(parameter.grad).all(), (overrides, name))

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


class V6MulticlassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        register_all_modules(init_default_scope=True)

    def test_public_configs(self):
        reference = Config.fromfile('configs/rpgv/v6/rpgv_v6.py')
        for dataset, classes in (('uavid', 8), ('loveda', 7)):
            cfg = Config.fromfile(f'configs/{dataset}/rpgv_v6.py')
            self.assertEqual(cfg.model.type, 'RPGVRemoteSensingV6')
            self.assertEqual(cfg.model.num_classes, classes)
            for field in ('geometry_channels', 'cluster_cfg', 'attention_cfg',
                          'fusion_location', 'contour_mode', 'fusion_scale',
                          'contrastive_loss_weight', 'rectification_loss_weight'):
                self.assertEqual(cfg.model[field], reference.model[field])
            self.assertEqual(cfg.default_hooks.checkpoint.save_best, 'mIoU')
            self.assertEqual(cfg.test_dataloader, cfg.val_dataloader)
            self.assertEqual(cfg.predict_dataset.pipeline, cfg.predict_pipeline)
            self.assertNotIn('LoadAnnotations', [s.type for s in cfg.predict_pipeline])
            for pipeline in (cfg.train_pipeline, cfg.test_pipeline, cfg.predict_pipeline):
                self.assertIn('LoadPseudoGeometry', [s.type for s in pipeline])
                self.assertEqual(pipeline[-1].type, 'PackSegInputs')
            cfg.model.rgb_encoder.init_cfg = None
            model = MODELS.build(cfg.model)
            self.assertIsInstance(model, RPGVRemoteSensingV6)
            self.assertEqual(model.refiner.sdf_head.out_channels, classes)

    def test_classes_losses_backward_and_reload(self):
        for classes in (7, 8):
            model = tiny_model(type='RPGVRemoteSensingV6', num_classes=classes)
            inputs, samples = batch()
            for sample in samples:
                sample.gt_sem_seg.data[:, :, :32] = 0
                sample.gt_sem_seg.data[:, :, 32:] = classes - 1
                sample.gt_sem_seg.data[:, :4] = 255
            target, valid = model._targets(samples, (64, 64), inputs.device)
            self.assertEqual(target.max().item(), classes - 1)
            self.assertFalse(valid[:, :, :4].any())
            outputs = model._run_network(inputs)
            for field in ('final_logits', 'coarse_logits', 'sdf', 'seg_logits'):
                self.assertEqual(outputs[field].shape[1], classes)
            losses = model.loss(inputs, samples)
            self.assertTrue({'loss_final', 'loss_coarse', 'loss_sdf', 'loss_boundary_band',
                             'loss_region_contrastive', 'loss_depth_rectification'} <= losses.keys())
            self.assertGreater(losses['loss_boundary_band'].item(), 0)
            self.assertTrue(torch.isfinite(objective(losses)))
            objective(losses).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            replica = tiny_model(type='RPGVRemoteSensingV6', num_classes=classes)
            replica.load_state_dict(model.state_dict(), strict=True)
            model.eval()
            replica.eval()
            with torch.no_grad():
                torch.testing.assert_close(model.encode_decode(inputs), replica.encode_decode(inputs))

    def test_high_class_id_semantic_supervision(self):
        model = tiny_model(type='RPGVRemoteSensingV6', num_classes=8)
        target = torch.full((1, 1, 8, 8), 7, dtype=torch.long)
        valid = torch.ones_like(target, dtype=torch.bool)
        correct = torch.zeros(1, 8, 8, 8)
        correct[:, 7] = 10
        incorrect = torch.zeros_like(correct)
        incorrect[:, 1] = 10
        self.assertLess(model._cross_entropy(correct, target, valid).item(), 0.001)
        self.assertGreater(model._cross_entropy(incorrect, target, valid).item(), 9)
        self.assertEqual(model._boundary_band_loss(correct, target, valid).item(), 0)

    def test_multiclass_ignore_and_uniform_labels(self):
        for label in (255, 0, 6):
            model = tiny_model(type='RPGVRemoteSensingV6', num_classes=7)
            inputs, samples = batch(label=label)
            losses = model.loss(inputs, samples)
            self.assertTrue(torch.isfinite(objective(losses)))
            self.assertEqual(losses['loss_boundary_band'].item(), 0)
            if label == 255:
                self.assertEqual(objective(losses).item(), 0)
            objective(losses).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_multiclass_fallback_whole_and_slide(self):
        for classes in (7, 8):
            model = tiny_model(type='RPGVRemoteSensingV6', num_classes=classes).eval()
            inputs, samples = batch((80, 83), batch_size=1)
            inputs[:, 4] = 0
            with torch.no_grad():
                expected = model._run_rgb_branch(inputs)['seg_logits']
                for value in (inputs, inputs[:, :3]):
                    self.assertTrue(torch.equal(model._run_network(value)['seg_logits'], expected))
                self.assertEqual(tuple(model.encode_decode(inputs).shape), (1, classes, 80, 83))
                model.test_cfg = dict(mode='slide', crop_size=(64, 64), stride=(48, 48))
                logits = model.inference(inputs, [s.metainfo for s in samples])
                self.assertEqual(tuple(logits.shape), (1, classes, 80, 83))
                self.assertTrue(torch.isfinite(logits).all())
                result = model.predict(inputs, samples)[0]
                self.assertEqual(tuple(result.pred_sem_seg.data.shape), (1, 80, 83))
                self.assertLess(result.pred_sem_seg.data.max().item(), classes)


if __name__ == '__main__':
    unittest.main()
