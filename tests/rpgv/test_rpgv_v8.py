"""Math, gradient and MMSeg contracts for RGB-only Gaussian RPGV v8."""
from __future__ import annotations

import copy
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401
from wwtpseg.models.segmentors.multiclass import RPGVRemoteSensingV8  # noqa: F401
from wwtpseg.models.utils.rpgv_v6_modules import _coordinates
from wwtpseg.models.utils.rpgv_v8_modules import (GaussianRelationLayerV8, GaussianRelationNetworkV8,
                                                 GaussianSplatV8, GaussianPrimitivePredictorV8,
                                                 bounded_grid, contact_descriptor)
from wwtpseg.models.utils.rpgv_v8_losses import (boundary_band_loss, class_proportions,
                                               gaussian_semantic_losses)
from wwtpseg.experiments import DATASETS, V8_ABLATIONS, experiment_config
from tests.rpgv.test_rpgv_v6 import batch as geometry_batch, objective


def tiny_model(**overrides):
    config = dict(
        type='RPGVNetV8',
        rgb_encoder=dict(type='MixVisionTransformerSDPA', in_channels=3, embed_dims=8,
                         num_stages=4, num_layers=[1, 1, 1, 1], num_heads=[1, 2, 4, 8],
                         patch_sizes=[7, 3, 3, 3], sr_ratios=[8, 4, 2, 1],
                         out_indices=(0, 1, 2, 3), drop_path_rate=0.0),
        rgb_channels=(8, 16, 32, 64), decoder_channels=16, detail_channels=8,
        gaussian_channels=16, gaussian_cfg=dict(grid_sizes=(2,), max_size=16),
        graph_cfg=dict(heads=4, layers=2))
    config.update(overrides)
    return MODELS.build(config)


def batch(**kwargs):
    images, samples = geometry_batch(**kwargs)
    return images[:, :3].contiguous(), samples


def pyramid(spatial=(256, 128), channels=(8, 16, 32, 64)):
    return [torch.randn(1, value, spatial[0] // (2 ** index), spatial[1] // (2 ** index))
            for index, value in enumerate(channels)]


def primitives(dtype=torch.float32):
    return dict(centers=torch.tensor([[[-0.4, 0.2], [0.3, -0.1]]], dtype=dtype),
                cholesky=torch.tensor([[[[0.4, 0.], [0.1, 0.3]],
                                         [[0.3, 0.], [-0.1, 0.5]]]], dtype=dtype),
                amplitude=torch.tensor([[[0.7], [0.9]]], dtype=dtype),
                height=torch.tensor([[[-0.2], [0.6]]], dtype=dtype))


class GaussianMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_kernel_matches_direct_covariance_reference(self):
        values = primitives(torch.float64)
        raster = GaussianSplatV8()(values, (5, 7))
        from wwtpseg.models.utils.rpgv_v6_modules import _coordinates
        xy = _coordinates(5, 7, values['centers']).flatten(2).transpose(1, 2)
        delta = xy[:, None] - values['centers'][:, :, None]
        covariance = values['cholesky'] @ values['cholesky'].transpose(-1, -2)
        quadratic = torch.einsum('bnpi,bnij,bnpj->bnp', delta, covariance.inverse(), delta)
        kernel = values['amplitude'] * (-0.5 * quadratic).exp()
        torch.testing.assert_close(raster['support'], kernel / kernel.sum(1, keepdim=True))
        torch.testing.assert_close(raster['support'].sum(1), torch.ones(1, 35, dtype=torch.float64))
        torch.testing.assert_close(raster['coverage'].flatten(2), -torch.expm1(-kernel.sum(1, keepdim=True)))
        self.assertTrue((raster['variance'] >= 0).all())

    def test_permutation_and_constant_height(self):
        values = primitives()
        features = torch.randn(1, 2, 3)
        splat = GaussianSplatV8()
        raster = splat(values, (7, 9))
        permutation = torch.tensor([1, 0])
        other = splat({key: value[:, permutation] for key, value in values.items()}, (7, 9))
        for key in ('height', 'coverage', 'variance'):
            torch.testing.assert_close(raster[key], other[key])
        torch.testing.assert_close(splat.render(features, raster), splat.render(features[:, permutation], other))
        values['height'].fill_(0.4)
        constant = splat(values, (1, 9))
        torch.testing.assert_close(constant['height'], torch.full((1, 1, 1, 9), 0.4))
        self.assertLess(constant['variance'].max().item(), 1e-12)

    def test_anisotropy_and_amplitude_affect_rendering(self):
        values = primitives()
        splat = GaussianSplatV8()
        initial = splat(values, (11, 13))
        rotated = {key: value.clone() for key, value in values.items()}
        rotated['cholesky'][..., 1, 0] *= -1
        self.assertFalse(torch.allclose(initial['support'], splat(rotated, (11, 13))['support']))
        values['amplitude'] *= 2
        changed = splat(values, (11, 13))
        torch.testing.assert_close(initial['support'], changed['support'])
        self.assertGreater((changed['coverage'] - initial['coverage']).min().item(), 0)

    def test_gradcheck(self):
        values = primitives(torch.float64)
        inputs = tuple(value.requires_grad_() for value in values.values())
        def render(*args):
            raster = GaussianSplatV8()(dict(zip(values, args)), (3, 4))
            return torch.cat([raster[key].flatten() for key in ('support', 'height', 'coverage', 'variance')])
        self.assertTrue(torch.autograd.gradcheck(render, inputs, eps=1e-6, atol=1e-4))

    def test_degenerate_and_extreme_inputs_remain_finite(self):
        for size in ((1, 1), (1, 7), (9, 1), (11, 13)):
            for scale, amplitude in ((1e-4, 0.), (1e3, 1e3)):
                values = primitives()
                values['cholesky'] = torch.eye(2)[None, None].repeat(1, 2, 1, 1) * scale
                values['centers'] += 10
                values['amplitude'].fill_(amplitude)
                for value in values.values():
                    value.requires_grad_()
                raster = GaussianSplatV8()(values, size)
                loss = sum(raster[key].square().mean() for key in ('height', 'coverage', 'variance', 'support'))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for value in values.values():
                    self.assertIsNotNone(value.grad)
                    self.assertTrue(torch.isfinite(value.grad).all())

    def test_bounded_grid_and_predictor_support(self):
        self.assertEqual(bounded_grid((256, 128), 64), (64, 32))
        predictor = GaussianPrimitivePredictorV8((8, 16, 32, 64), 16)
        self.assertEqual(predictor.grid_sizes, (4, 8, 16))
        values, raster, nodes = predictor(pyramid())
        self.assertEqual(raster['support'].shape, (1, 336, 64 * 32))
        covariance = values['cholesky'] @ values['cholesky'].transpose(-1, -2)
        self.assertTrue((torch.linalg.eigvalsh(covariance) > 0).all())
        self.assertEqual(nodes.shape, (1, 336, 16))
        self.assertTrue(raster['support'].requires_grad)

    def test_invalid_grid_and_contact_configurations_are_rejected(self):
        for values in ((0,), (8, -1), ()):
            with self.assertRaisesRegex(ValueError, 'grid_sizes'):
                GaussianPrimitivePredictorV8((8, 16, 32, 64), 16, grid_sizes=values)
        with self.assertRaisesRegex(ValueError, 'max_size'):
            GaussianPrimitivePredictorV8((8, 16, 32, 64), 16, max_size=0)
        with self.assertRaisesRegex(ValueError, 'at least one layer'):
            GaussianRelationNetworkV8(16, 4, layers=0)

    def test_multi_scale_anchor_counts_respect_the_raster_clamp(self):
        for spatial, maximum in (((256, 128), 64), ((256, 256), 64), ((32, 16), 16)):
            predictor = GaussianPrimitivePredictorV8((8, 16, 32, 64), 16, max_size=maximum)
            features = pyramid(spatial)
            raster = predictor(features)[1]
            size = bounded_grid(features[0].shape[-2:], maximum)
            expected = sum(min(value, size[0]) * min(value, size[1]) for value in (4, 8, 16))
            self.assertEqual(raster['support'].shape[:2], (1, expected), (spatial, maximum))
            self.assertEqual(raster['size'], size)

    def test_center_reach_scales_per_level_and_stays_bounded(self):
        predictor = GaussianPrimitivePredictorV8((8, 16, 32, 64), 16)
        features = pyramid()
        self.assertEqual(predictor.center_log_scale.item(), 0.0)
        values, _, _ = predictor(features)
        levels = ((4, 4), (8, 8), (16, 16))
        anchors = torch.cat([_coordinates(*grid, values['centers']).flatten(2).transpose(1, 2)
                             for grid in levels], dim=1)
        step = torch.cat([values['centers'].new_tensor((2.0 / grid[1], 2.0 / grid[0])).expand(
            grid[0] * grid[1], 2) for grid in levels], dim=0).unsqueeze(0)
        # Each level carries its own lattice pitch, coarse anchors first.
        self.assertEqual(step[0, :16, 0].unique().tolist(), [0.5])
        self.assertEqual(step[0, 16:80, 0].unique().tolist(), [0.25])
        self.assertEqual(step[0, 80:, 0].unique().tolist(), [0.125])
        offset = (values['centers'] - anchors) / step
        self.assertLessEqual(offset.abs().max().item(), 1.0)
        self.assertGreater(offset.abs().max().item(), 0.0)
        with torch.no_grad():
            predictor.center_log_scale.add_(math.log(2.0))
        widened = predictor(features)[0]
        torch.testing.assert_close(widened['centers'] - anchors, 2.0 * (values['centers'] - anchors),
                                   rtol=1e-6, atol=1e-7)
        self.assertLessEqual(((widened['centers'] - anchors) / step).abs().max().item(), 2.0)

    def test_contact_descriptor_matches_shifted_reference_and_keeps_gradient(self):
        support = torch.rand(2, 3, 5 * 4)
        support = support / support.sum(1, keepdim=True)
        support.requires_grad_()
        contact = contact_descriptor(support, (5, 4))
        maps = support.reshape(2, 3, 5, 4)
        reference = torch.zeros(2, 3, 3)
        for shift in ((0, 1), (1, 0)):
            left = maps[:, :, :5 - shift[0], :4 - shift[1]].reshape(2, 3, -1)
            right = maps[:, :, shift[0]:, shift[1]:].reshape(2, 3, -1)
            cross = torch.bmm(left, right.transpose(1, 2))
            reference = reference + cross + cross.transpose(1, 2)
        area = support.sum(-1)
        reference = reference / (area.unsqueeze(-1) * area.unsqueeze(-2)).sqrt()
        torch.testing.assert_close(contact[..., 0], reference)
        self.assertTrue((contact >= 0).all())
        self.assertLessEqual(contact.max().item(), 2.0)
        contact.sum().backward()
        self.assertIsNotNone(support.grad)
        self.assertTrue(torch.isfinite(support.grad).all())
        self.assertGreater(support.grad.abs().sum().item(), 0)

    def test_contact_descriptor_degenerate_raster_is_zero(self):
        support = torch.full((1, 2, 1), 0.5, requires_grad=True)
        contact = contact_descriptor(support, (1, 1))
        self.assertEqual(contact.shape, (1, 2, 2, 1))
        self.assertEqual(contact.abs().sum().item(), 0)

    def test_contact_dimension_reaches_the_edge_bias(self):
        layer = GaussianRelationLayerV8(16, 4, use_contact=True)
        self.assertEqual(layer.edge_bias[0].in_features, 9)
        self.assertEqual(GaussianRelationLayerV8(16, 4, use_contact=False).edge_bias[0].in_features, 8)
        nodes = torch.randn(2, 5, 16)
        edges = torch.randn(2, 5, 5, 9)
        baseline = layer(nodes, edges)
        perturbed = edges.clone()
        perturbed[..., -1] += 1.0
        with torch.no_grad():
            self.assertFalse(torch.allclose(baseline, layer(nodes, perturbed)))

    def test_valid_class_pooling_and_unknown_boundaries(self):
        target = torch.tensor([[[[0, 1], [0, 0]]]])
        valid = torch.tensor([[[[True, True], [False, False]]]])
        distribution, weight = class_proportions(target, valid, (1, 1), 2)
        torch.testing.assert_close(distribution.flatten(), torch.tensor([0.5, 0.5]))
        self.assertEqual(weight.item(), 0.5)
        self.assertFalse(distribution.requires_grad)
        # A change hidden entirely behind ignore must not create a semantic boundary.
        target = torch.zeros(1, 1, 9, 9, dtype=torch.long)
        target[..., 4:] = 1
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[..., 4:] = False
        logits = torch.randn(1, 1, 9, 9, requires_grad=True)
        loss = boundary_band_loss(logits, target, valid, True, 1)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)


class V8ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        register_all_modules()

    def setUp(self):
        torch.manual_seed(42)

    def test_final_loss_trains_each_gaussian_parameter_family(self):
        model = tiny_model()
        images, samples = batch()
        model.loss(images, samples)['loss_final'].backward()
        for name in ('center_head', 'covariance_head', 'amplitude_head', 'height_head', 'feature_head'):
            gradient = getattr(model.gaussian_predictor, name).weight.grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertTrue((gradient.abs().sum(dim=1) > 0).all(), name)
        # The reach scalar sees the centers through tanh(offset) * step, so a random
        # center_head must leave it a nonzero gradient; otherwise the mechanism is inert.
        reach = model.gaussian_predictor.center_log_scale.grad
        self.assertIsNotNone(reach)
        self.assertTrue(torch.isfinite(reach).all())
        self.assertNotEqual(reach.item(), 0.0)
        for name in ('gaussian_graph', 'structure_fusion', 'local_fusion'):
            gradients = [p.grad for p in getattr(model, name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, name)

    def test_full_loss_has_no_unused_parameters_or_depth_modules(self):
        model = tiny_model()
        images, samples = batch()
        losses = model.loss(images, samples)
        self.assertEqual(set(losses), {'loss_final', 'loss_coarse', 'loss_sdf', 'loss_boundary_band',
                                      'loss_gaussian_dense', 'loss_gaussian_node'})
        objective(losses).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for name in ('depth_rectifier', 'geometry_encoder', 'frequency_split', 'hierarchical', 'region_contrastive'):
            self.assertFalse(hasattr(model, name))
        with self.assertRaisesRegex(ValueError, 'three'):
            model.encode_decode(torch.rand(2, 5, 64, 64))

    def test_gaussian_intervention_changes_final_logits(self):
        model = tiny_model().eval()
        images, _ = batch()
        with torch.no_grad():
            before = model.encode_decode(images)
            model.gaussian_predictor.height_head.bias.add_(0.7)
            changed_height = model.encode_decode(images)
            model.gaussian_predictor.center_head.bias.add_(0.7)
            changed_center = model.encode_decode(images)
            model.gaussian_predictor.center_log_scale.add_(0.5)
            changed_reach = model.encode_decode(images)
        self.assertGreater((before - changed_height).abs().max().item(), 1e-6)
        self.assertGreater((changed_height - changed_center).abs().max().item(), 1e-6)
        self.assertGreater((changed_center - changed_reach).abs().max().item(), 1e-6)

    def test_auxiliary_targets_detach_but_support_keeps_gradients(self):
        model = tiny_model()
        images, samples = batch()
        outputs = model._run_network(images)
        support = outputs['gaussian']['raster']['support']
        support.retain_grad()
        target = torch.stack([sample.gt_sem_seg.data for sample in samples])
        objective(gaussian_semantic_losses(outputs, target, target != 255, True, 0.1, 0.05)).backward()
        self.assertIsNotNone(support.grad)
        self.assertGreater(support.grad.abs().sum().item(), 0)
        # Isolate target construction: independent logits cannot train the support.
        outputs = model._run_network(images)
        outputs['gaussian_node_logits'] = torch.randn(2, 4, 1, requires_grad=True)
        support = outputs['gaussian']['raster']['support']
        loss = gaussian_semantic_losses(outputs, target, target != 255, True, 0, 1)['loss_gaussian_node']
        self.assertIsNone(torch.autograd.grad(loss, support, allow_unused=True)[0])

    def test_binary_categorical_ignore_and_checkpoint_roundtrip(self):
        for classes in (None, 2, 7, 8):
            options = {} if classes is None else dict(type='RPGVRemoteSensingV8', num_classes=classes)
            model = tiny_model(**options)
            images, samples = batch(size=(65, 79))
            if classes:
                for sample in samples:
                    sample.gt_sem_seg.data.random_(0, classes)
            for sample in samples:
                sample.gt_sem_seg.data[..., :3] = 255
            losses = model.loss(images, samples)
            self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
            objective(losses).backward()
            model.eval()
            restored = tiny_model(**options).eval()
            restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
            with torch.no_grad():
                original = model.encode_decode(images)
                torch.testing.assert_close(original, restored.encode_decode(images), rtol=0, atol=0)
                predictions = model.predict(images, samples)
            self.assertEqual(original.shape, (2, classes or 2, 65, 79))
            self.assertEqual(predictions[0].pred_sem_seg.data.shape, (1, 65, 79))
            for sample in samples:
                sample.gt_sem_seg.data.fill_(255)
            model.zero_grad(set_to_none=True)
            ignored = model.loss(images, samples)
            self.assertTrue(all(value.item() == 0 and value.requires_grad for value in ignored.values()))
            objective(ignored).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    def test_ablation_modules_and_gradients(self):
        options = [dict(use_gaussians=False),
                   dict(gaussian_cfg=dict(grid_sizes=(2,), fixed_gaussians=True)),
                   dict(use_message_passing=False), dict(gaussian_dense_weight=0, gaussian_node_weight=0),
                   dict(gaussian_cfg=dict(grid_sizes=(2,), center_scale_learnable=False)),
                   dict(graph_cfg=dict(heads=4, layers=2, use_contact=False))]
        for values in options:
            model = tiny_model(**values)
            images, samples = batch(label=0)
            objective(model.loss(images, samples)).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, (values, name))
            if values.get('use_gaussians') is False:
                self.assertFalse(hasattr(model, 'gaussian_predictor'))
            gaussian = values.get('gaussian_cfg', {})
            if gaussian.get('fixed_gaussians'):
                self.assertFalse(hasattr(model.gaussian_predictor, 'center_head'))
                self.assertFalse(hasattr(model.gaussian_predictor, 'covariance_head'))
            if gaussian.get('center_scale_learnable') is False:
                # Frozen reach must be a buffer, never a parameter without a gradient.
                self.assertNotIn('center_log_scale', dict(model.gaussian_predictor.named_parameters()))
                self.assertIn('center_log_scale', model.gaussian_predictor.state_dict())
            if values.get('use_message_passing') is False:
                self.assertFalse(hasattr(model, 'gaussian_graph'))
            if 'gaussian_dense_weight' in values:
                self.assertFalse(hasattr(model, 'gaussian_dense_head'))
                self.assertFalse(hasattr(model, 'gaussian_node_head'))

    def test_ablation_state_dict_layouts_match(self):
        reference = set(tiny_model().state_dict())
        for values in (dict(gaussian_cfg=dict(grid_sizes=(2,), center_scale_learnable=False)),
                       dict(graph_cfg=dict(heads=4, layers=2, use_contact=False))):
            self.assertEqual(set(tiny_model(**values).state_dict()), reference, values)
        # Fixing the primitives removes the two shape heads outright, so that
        # variant intentionally cannot resume the default optimizer state.
        fixed = tiny_model(gaussian_cfg=dict(grid_sizes=(2,), fixed_gaussians=True))
        dropped = {name for name in reference
                   if name.startswith(('gaussian_predictor.center_head', 'gaussian_predictor.covariance_head'))}
        self.assertEqual(set(fixed.state_dict()), reference - dropped)
        self.assertIn('gaussian_predictor.center_log_scale', fixed.state_dict())

    def test_slide_prediction_covers_categorical_frame(self):
        model = tiny_model(type='RPGVRemoteSensingV8', num_classes=8,
                           test_cfg=dict(mode='slide', crop_size=(64, 64), stride=(32, 32))).eval()
        images, samples = batch(size=(81, 97), batch_size=1)
        with torch.no_grad():
            prediction = model.predict(images, samples)[0]
        self.assertEqual(prediction.seg_logits.data.shape, (8, 81, 97))
        self.assertTrue(torch.isfinite(prediction.seg_logits.data).all())
        self.assertLess(prediction.pred_sem_seg.data.max().item(), 8)

    def test_configs_rgb_only_and_shared_protocol(self):
        protocol = Config.fromfile('configs/_base_/schedules/baseline_20k.py')
        for dataset in DATASETS:
            cfg = Config.fromfile(experiment_config(dataset, 'rpgv_v8'))
            self.assertEqual(cfg.optim_wrapper, protocol.optim_wrapper)
            self.assertEqual(cfg.param_scheduler, protocol.param_scheduler)
            self.assertEqual(cfg.train_cfg, protocol.train_cfg)
            for split in ('train', 'val', 'test'):
                pipeline = cfg[f'{split}_dataloader'].dataset.pipeline
                self.assertNotIn('LoadPseudoGeometry', [step.type for step in pipeline])
            if 'predict_pipeline' in cfg:
                predict_steps = [step.type for step in cfg.predict_pipeline]
                self.assertNotIn('LoadPseudoGeometry', predict_steps)
                self.assertNotIn('LoadAnnotations', predict_steps)
            self.assertFalse(cfg.model.data_preprocessor.bgr_to_rgb)
            self.assertNotIn('mean', cfg.model.data_preprocessor)
        for name in V8_ABLATIONS:
            self.assertTrue(experiment_config('wwtp', name).is_file())

    def test_optimizer_covers_all_parameters_once(self):
        model = tiny_model()
        cfg = Config.fromfile('configs/_base_/schedules/baseline_20k.py')
        wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
        ids = [id(p) for group in wrapper.optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
        norm_ids = {id(p) for module in model.modules()
                    if isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm))
                    for p in module.parameters(recurse=False)}
        reach_group = None
        for group in wrapper.optimizer.param_groups:
            if any(id(p) in norm_ids for p in group['params']):
                self.assertEqual(group['weight_decay'], 0)
            if any(p is model.gaussian_predictor.center_log_scale for p in group['params']):
                reach_group = group
        # The reach scalar is a plain parameter and takes the shared 0.01 decay, so
        # the common protocol needs no exemption for the new mechanism.
        self.assertIsNotNone(reach_group)
        self.assertEqual(reach_group['weight_decay'], 0.01)

    def test_new_ablation_configs_change_one_factor_each(self):
        full = Config.fromfile('configs/rpgv/v8/rpgv_v8.py')
        self.assertEqual(tuple(full.model.gaussian_cfg.grid_sizes), (4, 8, 16))
        self.assertEqual(full.model.gaussian_cfg.max_size, 64)
        self.assertTrue(full.model.graph_cfg.use_contact)
        bounded = Config.fromfile('configs/rpgv/v8/bounded_centers.py')
        self.assertFalse(bounded.model.gaussian_cfg.center_scale_learnable)
        self.assertEqual(tuple(bounded.model.gaussian_cfg.grid_sizes), (4, 8, 16))
        self.assertEqual(bounded.model.gaussian_cfg.max_size, 64)
        self.assertTrue(bounded.model.graph_cfg.use_contact)
        no_contact = Config.fromfile('configs/rpgv/v8/no_contact.py')
        self.assertFalse(no_contact.model.graph_cfg.use_contact)
        self.assertEqual(tuple(no_contact.model.gaussian_cfg.grid_sizes), (4, 8, 16))
        dense = Config.fromfile('configs/rpgv/v8/single_scale_dense.py')
        self.assertEqual(tuple(dense.model.gaussian_cfg.grid_sizes), (20,))
        self.assertTrue(dense.model.graph_cfg.use_contact)
        fixed = Config.fromfile('configs/rpgv/v8/fixed_gaussians.py')
        self.assertTrue(fixed.model.gaussian_cfg.fixed_gaussians)
        self.assertEqual(tuple(fixed.model.gaussian_cfg.grid_sizes), (4, 8, 16))
        self.assertTrue(fixed.model.graph_cfg.use_contact)

    def test_single_scale_control_carries_more_anchors_than_the_default(self):
        # On the 1024x1024 WWTP raster both configs fill the 64x64 structure grid,
        # so the control's node count is strictly larger and a default win cannot
        # be attributed to capacity alone.
        def anchors(grid_sizes):
            return sum(min(value, 64) ** 2 for value in grid_sizes)

        full = Config.fromfile('configs/rpgv/v8/rpgv_v8.py')
        dense = Config.fromfile('configs/rpgv/v8/single_scale_dense.py')
        self.assertEqual(anchors(full.model.gaussian_cfg.grid_sizes), 336)
        self.assertEqual(anchors(dense.model.gaussian_cfg.grid_sizes), 400)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_autocast_update(self):
        model = tiny_model().cuda()
        images, samples = batch()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        before = model.gaussian_predictor.center_head.weight.detach().clone()
        with torch.autocast('cuda', dtype=torch.float16):
            loss = objective(model.loss(images.cuda(), samples))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.gaussian_predictor.center_head.weight))
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
