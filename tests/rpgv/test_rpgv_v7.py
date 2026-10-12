"""Numerical, structural and model regressions for the independent v7 path."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

from tests.rpgv.test_rpgv_v6 import tiny_model as v6_tiny_model, batch, objective
from wwtpseg.engine.runtime import disable_pretraining
from wwtpseg.experiments import (
    experiment_config, V7_ABLATIONS, V7_ANALYSIS_ABLATIONS, V7_LEGACY_ABLATIONS,
)
from wwtpseg.models.utils.rpgv_v6_modules import SmoothFrequencySplitV6
from wwtpseg.models.utils.rpgv_v7_modules import (
    GeometryPrimitivesV7, GeometryRelationNetworkV7, gather_nodes,
    relation_descriptors, spatial_neighbors, local_compositions, permute_composition,
)
from wwtpseg.models.utils.rpgv_v7_losses import MeasuredPairContrastV7, HierarchicalContrastV7


def tiny_model(**overrides):
    config = dict(type='RPGVNetV7', structure_channels=16,
                  primitive_cfg=dict(grid_size=2, max_size=16),
                  graph_cfg=dict(heads=4, neighbors=3),
                  hierarchical_cfg=dict(max_relations=16, composition_size=4))
    config.update(overrides)
    return v6_tiny_model(**config)


def tiny_from_config(cfg):
    return tiny_model(
        use_geometry=cfg.model.use_geometry,
        structure_source=cfg.model.structure_source,
        structure_feedback=cfg.model.structure_feedback,
        structure_aux_weight=cfg.model.structure_aux_weight,
        graph_cfg=dict(cfg.model.graph_cfg),
        hierarchical_cfg=dict(cfg.model.hierarchical_cfg),
        primitive_cfg=dict(grid_size=2, max_size=16,
                           fixed_grid=cfg.model.primitive_cfg.fixed_grid))


def primitive_batch(quality=1.0, flat=False, size=(9, 11), **kwargs):
    module = GeometryPrimitivesV7(channels=16, grid_size=2, max_size=16, **kwargs)
    depth = torch.full((2, 1, *size), 0.5) if flat else torch.rand(2, 1, *size)
    depth.requires_grad_()
    q = torch.full_like(depth, quality)
    low, high = SmoothFrequencySplitV6()(depth, q)
    result = module(depth, low, high, q, measured_depth=depth.detach())
    return module, depth, result


def local_example():
    return dict(nodes=torch.randn(1, 1, 3, 16),
                content=torch.tensor([[[[-2., .1, .1, 0., .1, 0.],
                                        [-1., .1, .1, 0., .4, 0.],
                                        [-3., .1, .1, 0., .8, 0.]]]]),
                positions=torch.tensor([[[[0., 0.], [0.3, 0.], [1., 0.]]]]),
                contact=torch.zeros(1, 1, 3, 3), valid=torch.ones(1, 1, 3, dtype=torch.bool))


class V7ModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_flat_and_low_confidence_nodes_train(self):
        for quality in (1., .02, 1e-4):
            with self.subTest(quality=quality):
                module, depth, result = primitive_batch(quality, flat=True)
                self.assertTrue(result['valid'].all())
                result['nodes'].square().mean().backward()
                self.assertGreater(module.encoder[0][0].weight.grad.abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(depth.grad).all())
                torch.testing.assert_close(result['assignment'].sum(1), torch.ones(2, 9, 11))

    def test_missing_depth_is_not_observed(self):
        module = GeometryPrimitivesV7(channels=16, grid_size=2)
        depth, q = torch.rand(2, 1, 9, 11), torch.ones(2, 1, 9, 11)
        q[..., :5] = 0
        changed = torch.where(q > 0, depth, torch.full_like(depth, 1e6))
        outputs = []
        for x in (depth, changed):
            low, high = SmoothFrequencySplitV6()(x, q)
            outputs.append(module(x, low, high, q, measured_depth=x))
        for name in outputs[0]:
            torch.testing.assert_close(outputs[0][name], outputs[1][name], rtol=0, atol=0)

    def test_empty_and_single_pixel_primitives(self):
        for size in ((9, 11), (1, 1), (1, 3)):
            _, _, result = primitive_batch(0., size=size)
            self.assertFalse(result['valid'].any())
            for value in result.values():
                self.assertTrue(torch.isfinite(value).all())
            graph = GeometryRelationNetworkV7(16, neighbors=3)
            nodes, _ = graph(result['nodes'], result['content'], result['positions'],
                             result['contact'], result['valid'])
            self.assertEqual(nodes.abs().sum().item(), 0)
            nodes.sum().backward()

    def test_detached_targets_and_pooling(self):
        module, depth, result = primitive_batch()
        for key in ('content', 'positions', 'contact', 'edges', 'pooling', 'valid'):
            self.assertFalse(result[key].requires_grad, key)
        rgb = torch.randn(2, 16, 9, 11, requires_grad=True)
        rgb_nodes = module.pool(rgb, result['pooling'])
        self.assertIsNone(torch.autograd.grad(rgb_nodes.sum(), depth, allow_unused=True)[0])
        gradient = torch.autograd.grad(result['contrast_nodes'].sum(), depth)[0]
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_fixed_grid_measurements_and_flip(self):
        _, _, result = primitive_batch(flat=True, fixed_grid=True, size=(8, 8))
        expected = torch.tensor([[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]])
        torch.testing.assert_close(result['positions'][0], expected)
        torch.testing.assert_close(result['content'][..., 0], torch.full((2, 4), -torch.log(torch.tensor(4.))))
        torch.testing.assert_close(result['content'][..., 4], torch.full((2, 4), .5))
        edges = result['edges']
        self.assertEqual(edges[0, 0, 1, 0].item(), 1.)
        self.assertGreater(edges[0, 0, 1, 3].item(), 0)
        self.assertEqual(edges[0, 0, 3, 3].item(), 0)
        flipped = result['positions'] * torch.tensor([-1., 1.])
        changed = relation_descriptors(result['content'], flipped, result['contact'])
        torch.testing.assert_close(changed[..., :4], edges[..., :4])
        torch.testing.assert_close(changed[..., 4], -edges[..., 4])
        torch.testing.assert_close(changed[..., 5], edges[..., 5])

    def test_uniform_quality_does_not_shrink_measured_area(self):
        module = GeometryPrimitivesV7(16, grid_size=2, fixed_grid=True)
        depth = torch.rand(2, 1, 8, 8)
        content = []
        for quality in (1., .01):
            q = torch.full_like(depth, quality)
            low, high = SmoothFrequencySplitV6()(depth, q)
            content.append(module(depth, low, high, q)['content'])
        torch.testing.assert_close(content[0], content[1], atol=1e-5, rtol=1e-5)

    def test_graph_relabeling_is_equivariant(self):
        _, _, p = primitive_batch()
        graph = GeometryRelationNetworkV7(16, neighbors=3)
        original, adjacency = graph(p['nodes'], p['content'], p['positions'], p['contact'], p['valid'])
        order = torch.tensor([2, 0, 3, 1])
        other, other_adj = graph(p['nodes'][:, order], p['content'][:, order], p['positions'][:, order],
                                 p['contact'][:, order][:, :, order], p['valid'][:, order])
        torch.testing.assert_close(other, original[:, order], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(other_adj, adjacency[:, order][:, :, order])
        self.assertLessEqual(adjacency.sum(-1).max(), 4)

    def test_no_message_passing_does_not_mix_node_features(self):
        _, _, p = primitive_batch()
        changed = p['nodes'].detach().clone()
        changed[:, 0] += torch.randn_like(changed[:, 0]) * 2
        for enabled in (True, False):
            with self.subTest(message_passing=enabled):
                graph = GeometryRelationNetworkV7(16, neighbors=3, use_edges=enabled).eval()
                with torch.no_grad():
                    original, adjacency = graph(p['nodes'], p['content'], p['positions'],
                                                p['contact'], p['valid'])
                    other, other_adjacency = graph(changed, p['content'], p['positions'],
                                                   p['contact'], p['valid'])
                torch.testing.assert_close(adjacency, other_adjacency)
                if enabled:
                    self.assertGreater((other[:, 1:] - original[:, 1:]).abs().sum().item(), 0)
                else:
                    torch.testing.assert_close(other[:, 1:], original[:, 1:], atol=0, rtol=0)

    def test_counterfactual_moves_content_not_positions(self):
        local = local_example()
        perm = torch.tensor([[[2, 0, 1]]])
        candidate, changed = permute_composition(local, perm)
        self.assertTrue(changed.all())
        torch.testing.assert_close(candidate['positions'], local['positions'])
        torch.testing.assert_close(candidate['contact'], local['contact'])
        torch.testing.assert_close(candidate['nodes'], local['nodes'][:, :, perm[0, 0]])
        torch.testing.assert_close(candidate['nodes'].sum(-2), local['nodes'].sum(-2))
        old_edges = relation_descriptors(local['content'], local['positions'], local['contact'])
        new_edges = relation_descriptors(candidate['content'], candidate['positions'], candidate['contact'])
        self.assertGreater((old_edges - new_edges).abs().sum().item(), 0)

    def test_identity_and_identical_swaps_are_not_negatives(self):
        local = local_example()
        _, changed = permute_composition(local, torch.tensor([[[0, 1, 2]]]))
        self.assertFalse(changed.any())
        local['content'] = local['content'][..., :1, :].expand_as(local['content'])
        _, changed = permute_composition(local, torch.tensor([[[2, 0, 1]]]))
        self.assertFalse(changed.any())

    def test_permutation_never_imports_invalid_content(self):
        local = local_example()
        local['valid'][..., 1] = False
        for _ in range(8):
            candidate, _ = permute_composition(local)
            torch.testing.assert_close(candidate['nodes'][..., 1, :], local['nodes'][..., 1, :])
            torch.testing.assert_close(candidate['nodes'][..., [0, 2], :].sum(-2),
                                       local['nodes'][..., [0, 2], :].sum(-2))

    def test_graph_detects_content_position_reassignment(self):
        local = local_example()
        changed, _ = permute_composition(local, torch.tensor([[[2, 0, 1]]]))
        def encode(graph, data):
            out, _ = graph(data['nodes'][0], data['content'][0], data['positions'][0],
                           data['contact'][0], data['valid'][0])
            return out.mean(-2)
        graph = GeometryRelationNetworkV7(16, neighbors=2)
        self.assertGreater((encode(graph, local) - encode(graph, changed)).abs().sum().item(), 1e-5)
        unordered = GeometryRelationNetworkV7(16, neighbors=2, use_edges=False)
        torch.testing.assert_close(encode(unordered, local), encode(unordered, changed))

    def test_pair_loss_matches_balanced_reference(self):
        module = MeasuredPairContrastV7(8)
        rgb, geo = torch.randn(3, 8), torch.randn(3, 8)
        descriptor = torch.tensor([[0.], [1.], [2.]])
        valid = torch.tensor([True, True, False])
        actual, count = module(rgb, geo, descriptor, valid)
        a, b = F.normalize(module.rgb(rgb), dim=-1), F.normalize(module.geometry(geo), dim=-1)
        logits = 10 * a @ b.T
        expected = F.softplus(-logits.diagonal()[:2]).mean() + F.softplus(torch.stack((logits[0, 1], logits[1, 0]))).mean()
        torch.testing.assert_close(actual, expected)
        self.assertEqual(count.item(), 2)
        empty, count = module(rgb, geo, descriptor, torch.zeros_like(valid))
        self.assertEqual(empty.item(), 0)
        empty.backward()
        self.assertTrue(all(p.grad is not None for p in module.parameters()))
        _, count = module(rgb, geo, torch.zeros_like(descriptor), torch.ones_like(valid))
        self.assertEqual(count.item(), 0)

    def test_composition_loss_matches_margin_reference(self):
        _, _, p = primitive_batch()
        graph = GeometryRelationNetworkV7(16, neighbors=3)
        _, adjacency = graph(p['nodes'], p['content'], p['positions'], p['contact'], p['valid'])
        objective = HierarchicalContrastV7(16, node_weight=0, relation_weight=0,
                                           composition_weight=.05, composition_size=4, negatives=2)
        # Identical candidates marked valid provide a deterministic zero-gap reference.
        with patch('wwtpseg.models.utils.rpgv_v7_losses.permute_composition',
                   side_effect=lambda local: (local, torch.ones(2, 4, dtype=torch.bool))):
            result = objective(p, torch.randn(2, 4, 16), graph, adjacency)
        torch.testing.assert_close(result['loss_composition_v7'], .05 * F.softplus(torch.tensor(.2)))
        self.assertEqual(result['composition_count_v7'].item(), 8)
        self.assertEqual(result['changed_candidate_count_v7'].item(), 16)


class V7ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_full_backward_all_parameters_and_optimizer_step(self):
        model = tiny_model()
        inputs, samples = batch()
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
        before = model.relation_graph.layers[0].qkv.weight.detach().clone()
        losses = model.loss(inputs, samples)
        parsed, _ = model.parse_losses(losses)
        torch.testing.assert_close(parsed, objective(losses))
        for value in losses.values():
            self.assertTrue(torch.isfinite(value).all())
        objective(losses).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.relation_graph.layers[0].qkv.weight))
        self.assertFalse(hasattr(model, 'region_contrastive'))

    def test_segmentation_alone_trains_structure(self):
        model = tiny_model()
        inputs, samples = batch()
        model.loss(inputs, samples)['loss_final'].backward()
        for module in (model.primitives.encoder, model.relation_graph, model.structure_fusion):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None), 0)

    def test_disabled_feedback_has_no_structure_to_final_path(self):
        model = tiny_model(structure_feedback=False)
        inputs, samples = batch()
        model.loss(inputs, samples)['loss_final'].backward()
        self.assertTrue(all(p.grad is None for p in model.relation_graph.parameters()))
        self.assertTrue(all(p.grad is None for p in model.primitives.parameters()))
        model.zero_grad()
        objective(model.loss(inputs, samples)).backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.relation_graph.parameters()), 0)

    def test_rgb_geometry_references_are_separate(self):
        model = tiny_model().eval()
        inputs, _ = batch()
        with torch.no_grad():
            first = model._run_network(inputs)['structure_v7']
            changed_rgb = inputs.clone()
            changed_rgb[:, :3] = torch.rand_like(inputs[:, :3]) * 255
            second = model._run_network(changed_rgb)['structure_v7']
            for key in first['primitives']:
                torch.testing.assert_close(first['primitives'][key], second['primitives'][key], rtol=0, atol=0)
            changed_geo = inputs.clone()
            changed_geo[:, 3] = torch.rand_like(inputs[:, 3]) * 255
            third = model._run_network(changed_geo)['structure_v7']
            for a, b in zip(first['rgb_features'], third['rgb_features']):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_empty_geometry_exact_rgb_fallback_after_update(self):
        model = tiny_model()
        x, samples = batch(size=(65, 67))
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
        objective(model.loss(x, samples)).backward()
        optimizer.step()
        model.eval()
        x[:, 4] = 0
        with torch.no_grad():
            actual = model._run_network(x)['final_logits']
            expected = model._run_rgb_branch(x)['final_logits']
            missing = model._run_network(x[:, :3])['final_logits']
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(missing, expected, rtol=0, atol=0)

    def test_all_ignore_empty_geometry_and_single_node(self):
        for case in ('ignore', 'empty', 'single'):
            model = tiny_model(primitive_cfg=dict(grid_size=1 if case == 'single' else 2, max_size=16))
            inputs, samples = batch(label=255 if case == 'ignore' else None)
            if case == 'empty':
                inputs[:, 4] = 0
            losses = model.loss(inputs, samples)
            objective(losses).backward()
            for name, value in losses.items():
                self.assertTrue(torch.isfinite(value), (case, name))
                if case == 'ignore' and 'loss' in name:
                    self.assertEqual(value.item(), 0, name)
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, (case, name))
                self.assertTrue(torch.isfinite(p.grad).all(), (case, name))
            self.assertEqual(losses['loss_composition_v7'].item(), 0)

    def test_mixed_support_does_not_leak_between_images(self):
        model = tiny_model().eval()
        inputs, _ = batch()
        inputs[0, 4] = 0
        with torch.no_grad():
            mixed = model._run_network(inputs)['final_logits'][0]
            rgb = model._run_rgb_branch(inputs[:1])['final_logits'][0]
        torch.testing.assert_close(mixed, rgb, atol=1e-5, rtol=1e-5)

    def test_rgb_capacity_ignores_external_depth_and_quality(self):
        model = tiny_model(structure_source='rgb').eval()
        x, _ = batch()
        other = x.clone()
        other[:, 3:] = 0
        with torch.no_grad():
            original = model._run_network(x)['final_logits']
            changed = model._run_network(other)['final_logits']
            missing = model._run_network(x[:, :3])['final_logits']
        torch.testing.assert_close(original, changed, rtol=0, atol=0)
        torch.testing.assert_close(original, missing, rtol=0, atol=0)

    def test_multiclass_loss_predictions_and_reload(self):
        model = tiny_model(type='RPGVRemoteSensingV7', num_classes=7)
        x, samples = batch(size=(65, 67))
        samples[0].gt_sem_seg.data[..., :10] = 6
        samples[1].gt_sem_seg.data[..., :10] = 255
        losses = model.loss(x, samples)
        objective(losses).backward()
        self.assertEqual(model.structure_head.out_channels, 7)
        clone = tiny_model(type='RPGVRemoteSensingV7', num_classes=7)
        clone.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        clone.eval()
        with torch.no_grad():
            a, b = model._run_network(x), clone._run_network(x)
            torch.testing.assert_close(a['final_logits'], b['final_logits'], rtol=0, atol=0)
            predictions = model.predict(x, samples)
        self.assertEqual(predictions[0].seg_logits.data.shape[0], 7)
        model.test_cfg = dict(mode='slide', crop_size=(64, 64), stride=(32, 32))
        with torch.no_grad():
            prediction = model.predict(x[:1], samples[:1])
        self.assertEqual(prediction[0].seg_logits.data.shape, (7, 65, 67))

    def test_disabled_objectives_do_not_generate_candidates(self):
        model = tiny_model(hierarchical_cfg=dict(node_weight=.05, relation_weight=0, composition_weight=0))
        x, samples = batch()
        with patch('wwtpseg.models.utils.rpgv_v7_losses.permute_composition', side_effect=AssertionError):
            losses = model.loss(x, samples)
        self.assertNotIn('loss_composition_v7', losses)
        self.assertNotIn('loss_relation_v7', losses)

    def test_core_and_analysis_ablation_config_differences(self):
        reference = Config.fromfile(experiment_config('wwtp', 'rpgv_v7'))
        changes = {
            'rgb_only': {('use_geometry',): False},
            'fixed_grid': {('primitive_cfg', 'fixed_grid'): True},
            'no_feedback': {('structure_feedback',): False},
            'no_hierarchical_supervision': {
                ('hierarchical_cfg', weight): 0.0
                for weight in ('node_weight', 'relation_weight', 'composition_weight')},
            'no_node_supervision': {('hierarchical_cfg', 'node_weight'): 0.0},
            'no_relation_supervision': {('hierarchical_cfg', 'relation_weight'): 0.0},
            'no_composition': {('hierarchical_cfg', 'composition_weight'): 0.0},
            'no_structure_aux': {('structure_aux_weight',): 0.0},
            'no_message_passing': {
                ('hierarchical_cfg', 'composition_weight'): 0.0,
                ('graph_cfg', 'use_edges'): False},
        }
        self.assertEqual(set(V7_ABLATIONS + V7_ANALYSIS_ABLATIONS),
                         {'rpgv_v7_' + suffix for suffix in changes})
        directories = {reference.work_dir}
        for suffix, overrides in changes.items():
            with self.subTest(variant=suffix):
                cfg = Config.fromfile(experiment_config('wwtp', 'rpgv_v7_' + suffix))
                expected = copy.deepcopy(reference.model)
                for path, value in overrides.items():
                    target = expected
                    for field in path[:-1]:
                        target = target[field]
                    target[path[-1]] = value
                self.assertEqual(cfg.model, expected)
                self.assertNotIn(cfg.work_dir, directories)
                directories.add(cfg.work_dir)
                if suffix != 'rgb_only':
                    for split in ('train', 'val', 'test'):
                        self.assertEqual(cfg[f'{split}_dataloader'], reference[f'{split}_dataloader'])
        paired = Config.fromfile(experiment_config('wwtp', 'rpgv_v7_no_composition'))
        expected = copy.deepcopy(paired.model)
        expected.graph_cfg.use_edges = False
        message = Config.fromfile(experiment_config('wwtp', 'rpgv_v7_no_message_passing'))
        self.assertEqual(message.model, expected)
        for current, historical in (('no_hierarchical_supervision', 'no_contrastive'),
                                    ('no_composition', 'no_perturbation')):
            new = Config.fromfile(experiment_config('wwtp', 'rpgv_v7_' + current))
            old = Config.fromfile(experiment_config('wwtp', 'rpgv_v7_' + historical))
            self.assertEqual(new.model, old.model)
            self.assertNotEqual(new.work_dir, old.work_dir)

    def test_core_and_analysis_ablation_objectives_and_heads(self):
        structure_keys = {'loss_node_v7', 'loss_relation_v7', 'loss_composition_v7',
                          'loss_structure_aux_v7'}
        removed = {
            'rpgv_v7': set(),
            'rpgv_v7_rgb_only': structure_keys,
            'rpgv_v7_fixed_grid': set(),
            'rpgv_v7_no_feedback': set(),
            'rpgv_v7_no_hierarchical_supervision': structure_keys - {'loss_structure_aux_v7'},
            'rpgv_v7_no_node_supervision': {'loss_node_v7'},
            'rpgv_v7_no_relation_supervision': {'loss_relation_v7'},
            'rpgv_v7_no_composition': {'loss_composition_v7'},
            'rpgv_v7_no_structure_aux': {'loss_structure_aux_v7'},
            'rpgv_v7_no_message_passing': {'loss_composition_v7'},
        }
        x, samples = batch()
        for name, absent in removed.items():
            with self.subTest(variant=name):
                cfg = Config.fromfile(experiment_config('wwtp', name))
                model = tiny_from_config(cfg)
                with patch('wwtpseg.models.utils.rpgv_v7_losses.permute_composition',
                           wraps=permute_composition) as generate:
                    losses = model.loss(x[:, :3] if not cfg.model.use_geometry else x, samples)
                self.assertEqual(structure_keys & losses.keys(), structure_keys - absent)
                composition = 'loss_composition_v7' not in absent
                self.assertEqual(generate.call_count, cfg.model.hierarchical_cfg.negatives if composition else 0)
                if not cfg.model.use_geometry:
                    self.assertFalse(hasattr(model, 'primitives'))
                    self.assertFalse(hasattr(model, 'relation_graph'))
                    continue
                self.assertEqual(hasattr(model, 'structure_head'), 'loss_structure_aux_v7' not in absent)
                self.assertEqual(hasattr(model, 'structure_fusion'), cfg.model.structure_feedback)
                self.assertEqual(hasattr(model, 'structure_rgb_projections'),
                                 bool((structure_keys - {'loss_structure_aux_v7'}) - absent))
                for weight, head in (('node_weight', 'node_contrast'),
                                     ('relation_weight', 'rgb_relation'),
                                     ('relation_weight', 'relation_prediction'),
                                     ('composition_weight', 'rgb_composition'),
                                     ('composition_weight', 'geometry_composition')):
                    self.assertEqual(hasattr(model.hierarchical, head),
                                     bool(cfg.model.hierarchical_cfg[weight]), head)
                self.assertEqual(model.relation_graph.use_edges, cfg.model.graph_cfg.use_edges)
                self.assertFalse(hasattr(model, 'region_contrastive'))

    def test_message_passing_pair_has_trainable_graph_and_relation_heads(self):
        x, samples = batch()
        for name in ('rpgv_v7_no_composition', 'rpgv_v7_no_message_passing'):
            with self.subTest(variant=name):
                model = tiny_from_config(Config.fromfile(experiment_config('wwtp', name)))
                for key in ('loss_final', 'loss_structure_aux_v7'):
                    model.zero_grad(set_to_none=True)
                    model.loss(x, samples)[key].backward()
                    gradient = sum(p.grad.abs().sum().item() for p in model.relation_graph.parameters()
                                   if p.grad is not None)
                    self.assertGreater(gradient, 0, key)
                model.zero_grad(set_to_none=True)
                model.loss(x, samples)['loss_relation_v7'].backward()
                self.assertTrue(all(p.grad is None for p in model.relation_graph.parameters()))
                for module in (model.hierarchical.rgb_relation, model.hierarchical.relation_prediction,
                               model.primitives.encoder):
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()
                                           if p.grad is not None), 0)
                model.zero_grad(set_to_none=True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
                parameter = next(model.relation_graph.layers[0].parameters())
                before = parameter.detach().clone()
                objective(model.loss(x, samples)).backward()
                optimizer.step()
                self.assertFalse(torch.equal(before, parameter))

    def test_removing_auxiliary_head_preserves_inference(self):
        full = tiny_model().eval()
        cfg = Config.fromfile(experiment_config('wwtp', 'rpgv_v7_no_structure_aux'))
        without = tiny_from_config(cfg).eval()
        without.load_state_dict({key: value for key, value in full.state_dict().items()
                                 if not key.startswith('structure_head.')}, strict=True)
        x, _ = batch()
        with torch.no_grad():
            expected = full._run_network(x)['final_logits']
            actual = without._run_network(x)['final_logits']
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_configs_and_shared_protocol(self):
        reference = Config.fromfile(experiment_config('wwtp', 'hrnet'))
        for dataset, names in (('wwtp', ('rpgv_v7',) + V7_ABLATIONS + V7_ANALYSIS_ABLATIONS + V7_LEGACY_ABLATIONS),
                               ('loveda', ('rpgv_v7',)), ('uavid', ('rpgv_v7',))):
            for name in names:
                with self.subTest(dataset=dataset, model=name):
                    cfg = Config.fromfile(experiment_config(dataset, name))
                    for field in ('optim_wrapper', 'param_scheduler', 'train_cfg', 'auto_scale_lr', 'randomness', 'env_cfg'):
                        self.assertEqual(cfg[field], reference[field], field)
                    self.assertIsNone(cfg.load_from)
                    self.assertFalse(cfg.resume)
                    disable_pretraining(cfg.model)
                    model = MODELS.build(cfg.model)
                    self.assertEqual(model.refiner.num_classes, {'wwtp': 1, 'loveda': 7, 'uavid': 8}[dataset])
                    del model

    def test_ablation_backward_paths_and_rgb_pipelines(self):
        x, samples = batch()
        for name in V7_ABLATIONS + V7_ANALYSIS_ABLATIONS + V7_LEGACY_ABLATIONS:
            with self.subTest(variant=name):
                cfg = Config.fromfile(experiment_config('wwtp', name))
                model = tiny_from_config(cfg)
                rgb_only_input = name in ('rpgv_v7_rgb_only', 'rpgv_v7_rgb_capacity')
                if rgb_only_input:
                    for split in ('train', 'val', 'test'):
                        self.assertNotIn('LoadPseudoGeometry', [s.type for s in cfg[f'{split}_dataloader'].dataset.pipeline])
                losses = model.loss(x[:, :3] if rgb_only_input else x, samples)
                parsed, _ = model.parse_losses(losses)
                parsed.backward()
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)

    def test_multiclass_all_ignore_is_zero(self):
        model = tiny_model(type='RPGVRemoteSensingV7', num_classes=8)
        inputs, samples = batch(label=255)
        losses = model.loss(inputs, samples)
        parsed, _ = model.parse_losses(losses)
        self.assertEqual(parsed.item(), 0)
        parsed.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_optimizer_covers_new_parameters_with_uniform_rules(self):
        cfg = Config.fromfile(experiment_config('wwtp', 'rpgv_v7'))
        model = tiny_model()
        wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
        norms = {id(p) for m in model.modules() if isinstance(m, (torch.nn.LayerNorm, torch.nn.GroupNorm,
                                                                 torch.nn.modules.batchnorm._BatchNorm))
                 for p in m.parameters(recurse=False)}
        covered = set()
        for group in wrapper.optimizer.param_groups:
            self.assertEqual(group['lr'], 6e-4)
            for parameter in group['params']:
                covered.add(id(parameter))
                self.assertEqual(group['weight_decay'], 0 if id(parameter) in norms else .01)
        self.assertEqual(covered, {id(p) for p in model.parameters()})


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class V7CUDATests(unittest.TestCase):
    def test_amp_empty_and_supported_gradients(self):
        register_all_modules(init_default_scope=True)
        for empty in (False, True):
            model = tiny_model().cuda()
            inputs, samples = batch()
            if empty:
                inputs[:, 4] = 0
            with torch.autocast('cuda', dtype=torch.float16):
                losses = model.loss(inputs.cuda(), samples)
                total = objective(losses)
            total.backward()
            self.assertTrue(torch.isfinite(total))
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == '__main__':
    unittest.main()
