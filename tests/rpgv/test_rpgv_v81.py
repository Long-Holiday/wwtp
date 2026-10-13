"""Numerical, gradient and inference contracts for RGB-only RPGV v8.1."""
import copy
import math
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F
from mmseg.utils import register_all_modules

from tests.rpgv.test_rpgv_v6 import objective
from tests.rpgv.test_rpgv_v8 import batch, primitives, pyramid, tiny_model as v8_model
from wwtpseg.models.segmentors.rpgv_net_v51 import RPGVNetV51
from wwtpseg.models.utils.rpgv_v8_losses import boundary_band_loss
from wwtpseg.models.utils.rpgv_v8_modules import GaussianPrimitivePredictorV8, GaussianSplatV8
from wwtpseg.models.utils.rpgv_v81_losses import (balanced_boundary_band_loss,
                                                final_pair_losses, gaussian_dense_dice_loss)
from wwtpseg.models.utils.rpgv_v81_modules import (ContentGaussianSplatV81,
                                                 GaussianPrimitivePredictorV81,
                                                 GaussianRelationLayerV81,
                                                 LocalGaussianFusionV81)


def tiny_model(**kwargs):
    kwargs.setdefault('type', 'RPGVNetV81')
    return v8_model(**kwargs)


class V81MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_content_support_matches_reference_and_preserves_coverage(self):
        values = primitives(torch.float64)
        logits = torch.randn(1, 2, 35, dtype=torch.float64)
        base = GaussianSplatV8()(values, (5, 7))
        result = ContentGaussianSplatV81()(values, (5, 7), logits)
        expected = base['support'] * logits.exp()
        expected = expected / expected.sum(1, keepdim=True)
        torch.testing.assert_close(result['support'], expected)
        torch.testing.assert_close(result['coverage'], base['coverage'])
        torch.testing.assert_close(result['height'].flatten(2),
                                   torch.bmm(values['height'].transpose(1, 2), expected))
        self.assertFalse(torch.allclose(result['support'], base['support']))
        zero = ContentGaussianSplatV81()(values, (5, 7), torch.zeros_like(logits))
        for key in ('support', 'height', 'coverage', 'variance'):
            torch.testing.assert_close(zero[key], base[key])

    def test_content_support_is_permutation_equivariant(self):
        values = primitives()
        logits = torch.randn(1, 2, 35)
        permutation = torch.tensor([1, 0])
        splat = ContentGaussianSplatV81()
        initial = splat(values, (5, 7), logits)
        changed = splat({key: value[:, permutation] for key, value in values.items()},
                        (5, 7), logits[:, permutation])
        torch.testing.assert_close(initial['support'][:, permutation], changed['support'])
        for key in ('height', 'coverage', 'variance'):
            torch.testing.assert_close(initial[key], changed[key])

    def test_content_splat_gradcheck(self):
        values = primitives(torch.float64)
        content = torch.randn(1, 2, 12, dtype=torch.float64, requires_grad=True)
        inputs = tuple(value.requires_grad_() for value in values.values()) + (content,)
        for spatial_support in (True, False):
            with self.subTest(spatial_support=spatial_support):
                def render(*args):
                    result = ContentGaussianSplatV81()(dict(zip(values, args[:-1])), (3, 4), args[-1],
                                                       spatial_support=spatial_support)
                    return torch.cat([result[key].flatten() for key in ('support', 'height', 'variance', 'coverage')])
                self.assertTrue(torch.autograd.gradcheck(render, inputs, eps=1e-6, atol=1e-4))

    def test_predictor_switch_is_exact_v8_and_has_no_content_parameters(self):
        before = GaussianPrimitivePredictorV8((8, 16, 32, 64), channels=16, grid_sizes=(2,), max_size=16)
        after = GaussianPrimitivePredictorV81((8, 16, 32, 64), channels=16,
                                              grid_sizes=(2,), max_size=16, content_aware=False)
        after.load_state_dict(before.state_dict(), strict=True)
        features = pyramid((16, 16))
        left, right = before(features), after(features)
        torch.testing.assert_close(left[2], right[2])
        torch.testing.assert_close(left[1]['support'], right[1]['support'])
        self.assertFalse(hasattr(after, 'content_query'))

    def test_predictor_multiscale_shape_and_content_gradients(self):
        predictor = GaussianPrimitivePredictorV81((8, 16, 32, 64), channels=16,
                                                  grid_sizes=(2, 4), max_size=16)
        _, raster, nodes = predictor(pyramid((16, 8)))
        self.assertEqual(raster['support'].shape, (1, 20, 128))
        self.assertEqual(raster['size'], (16, 8))
        torch.testing.assert_close(raster['support'].sum(1), torch.ones(1, 128))
        loss = (nodes * torch.randn_like(nodes)).sum() + (raster['support'] * torch.randn_like(raster['support'])).sum()
        loss.backward()
        for head in (predictor.content_query, predictor.content_key, predictor.center_head,
                     predictor.covariance_head):
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)

    def test_content_only_support_matches_reference_and_preserves_coverage(self):
        values = primitives(torch.float64)
        logits = torch.randn(1, 2, 35, dtype=torch.float64)
        splat = ContentGaussianSplatV81()
        hybrid = splat(values, (5, 7), logits)
        result = splat(values, (5, 7), logits, spatial_support=False)
        expected = logits.softmax(1)
        torch.testing.assert_close(result['support'], expected, rtol=0, atol=0)
        torch.testing.assert_close(result['coverage'], hybrid['coverage'], rtol=0, atol=0)
        torch.testing.assert_close(result['height'].flatten(2),
                                   torch.bmm(values['height'].transpose(1, 2), expected))
        changed_values = {key: value.clone() for key, value in values.items()}
        changed_values['centers'] += 0.3
        changed_values['cholesky'] *= 1.5
        changed_values['amplitude'] *= 2
        changed = splat(changed_values, (5, 7), logits, spatial_support=False)
        torch.testing.assert_close(result['support'], changed['support'], rtol=0, atol=0)
        self.assertFalse(torch.allclose(result['coverage'], changed['coverage']))
        self.assertFalse(torch.allclose(result['support'], hybrid['support']))

    def test_content_only_support_is_permutation_equivariant(self):
        values = primitives()
        logits = torch.randn(1, 2, 35)
        permutation = torch.tensor([1, 0])
        splat = ContentGaussianSplatV81()
        initial = splat(values, (5, 7), logits, spatial_support=False)
        changed = splat({key: value[:, permutation] for key, value in values.items()},
                        (5, 7), logits[:, permutation], spatial_support=False)
        torch.testing.assert_close(initial['support'][:, permutation], changed['support'])
        for key in ('height', 'coverage', 'variance'):
            torch.testing.assert_close(initial[key], changed[key])

    def test_missing_support_sources_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least one'):
            GaussianPrimitivePredictorV81((8, 16, 32, 64), channels=16,
                                          content_aware=False, spatial_support=False)
        with self.assertRaisesRegex(ValueError, 'content logits'):
            ContentGaussianSplatV81()(primitives(), (5, 7), spatial_support=False)

    def test_spatial_support_default_and_checkpoint_layout_unchanged(self):
        options = dict(channels=16, grid_sizes=(2,), max_size=16)
        default = GaussianPrimitivePredictorV81((8, 16, 32, 64), **options)
        explicit = GaussianPrimitivePredictorV81((8, 16, 32, 64), spatial_support=True, **options)
        content_only = GaussianPrimitivePredictorV81((8, 16, 32, 64), spatial_support=False, **options)
        explicit.load_state_dict(default.state_dict(), strict=True)
        content_only.load_state_dict(default.state_dict(), strict=True)
        features = pyramid((16, 16))
        left, right = default(features), explicit(features)
        for key in ('support', 'height', 'coverage', 'variance'):
            torch.testing.assert_close(left[1][key], right[1][key], rtol=0, atol=0)
        torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)
        self.assertEqual(set(default.state_dict()), set(content_only.state_dict()))

    def test_core_scale_controls_node_counts(self):
        for scales, count in (((4, 8, 16), 336), ((20,), 400)):
            with self.subTest(scales=scales):
                predictor = GaussianPrimitivePredictorV81((8, 16, 32, 64), channels=16,
                                                          grid_sizes=scales, max_size=24)
                _, raster, nodes = predictor(pyramid((24, 24)))
                self.assertEqual(nodes.shape[1], count)
                self.assertEqual(raster['support'].shape, (1, count, 576))

    def test_relation_aggregation_matches_explicit_edge_value_and_gradients(self):
        layer = GaussianRelationLayerV81(16, 4).double()
        nodes = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
        edges = torch.randn(2, 5, 5, 9, dtype=torch.float64, requires_grad=True)
        actual = layer(nodes, edges)
        qkv = layer.qkv(layer.norm(nodes)).reshape(2, 5, 3, 4, 4)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (query @ key.transpose(-1, -2) / 2 + layer.edge_bias(edges).permute(0, 3, 1, 2)).softmax(-1)
        edge_values = layer.edge_value(edges).reshape(2, 5, 5, 4, 4)
        message = attention @ value + torch.einsum('bhij,bijhc->bhic', attention, edge_values)
        expected = nodes + layer.output(message.transpose(1, 2).reshape(2, 5, 16))
        expected = expected + layer.ffn(expected)
        torch.testing.assert_close(actual, expected)
        probe = torch.randn_like(actual)
        args = (nodes, edges, layer.edge_value.weight)
        a = torch.autograd.grad((actual * probe).sum(), args, retain_graph=True)
        b = torch.autograd.grad((expected * probe).sum(), args)
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right)

    def test_relation_node_permutation(self):
        layer = GaussianRelationLayerV81(16, 4)
        nodes, edges = torch.randn(1, 5, 16), torch.randn(1, 5, 5, 9)
        permutation = torch.tensor([3, 1, 4, 0, 2])
        torch.testing.assert_close(layer(nodes, edges)[:, permutation],
                                   layer(nodes[:, permutation], edges[:, permutation][:, :, permutation]))

    def test_local_fusion_uses_high_resolution_rgb(self):
        fusion = LocalGaussianFusionV81(8, 16, 19)
        feature = torch.randn(2, 16, 16, 24, requires_grad=True)
        gaussian = torch.randn(2, 19, 4, 6, requires_grad=True)
        rgb = torch.randn(2, 8, 16, 24, requires_grad=True)
        output = fusion(feature, gaussian, rgb)
        self.assertEqual(output.shape, feature.shape)
        self.assertFalse(torch.allclose(output, fusion(feature, gaussian, rgb.flip(-1))))
        output.square().mean().backward()
        for value in (feature, gaussian, rgb):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_balanced_band_matches_v7_and_differs_from_v8(self):
        target = torch.zeros(1, 1, 13, 17, dtype=torch.long)
        target[..., 6, 4:13] = 1
        valid = torch.ones_like(target, dtype=torch.bool)
        logits = torch.full(target.shape, -2.)
        actual = balanced_boundary_band_loss(logits, target, valid, True, 3)
        expected = RPGVNetV51._boundary_band_loss(SimpleNamespace(boundary_band_radius=3), logits, target, valid)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(actual.item(), boundary_band_loss(logits, target, valid, True, 3).item())
        for label in (0, 1):
            self.assertEqual(balanced_boundary_band_loss(logits, torch.full_like(target, label), valid, True, 3).item(), 0)

    def test_categorical_band_unchanged(self):
        logits = torch.randn(2, 3, 9, 11)
        target = torch.randint(0, 3, (2, 1, 9, 11))
        valid = torch.rand_like(target.float()) > .1
        torch.testing.assert_close(balanced_boundary_band_loss(logits, target, valid, False, 1),
                                   boundary_band_loss(logits, target, valid, False, 1))

    def test_pair_losses_preserve_signed_boundary_and_ignore(self):
        target = torch.zeros(1, 1, 4, 6, dtype=torch.long)
        target[..., 3:] = 1
        valid = torch.ones_like(target, dtype=torch.bool)
        logits = (target.float() * 2 - 1) * 10
        good = final_pair_losses(logits, target, valid, True)
        bad = final_pair_losses(-logits, target, valid, True)
        self.assertEqual(good[0].item(), 0)
        self.assertLess(good[1].item(), bad[1].item())
        categorical = torch.cat((-logits / 2, logits / 2), dim=1)
        for left, right in zip(good, final_pair_losses(categorical, target, valid, False)):
            torch.testing.assert_close(left, right)
        for binary, output in ((True, logits), (False, categorical)):
            for loss in final_pair_losses(output, target, torch.zeros_like(valid), binary):
                self.assertEqual(loss.item(), 0)

    def test_dense_dice_uses_soft_proportions_and_valid_fraction(self):
        logits = torch.zeros(1, 1, 1, 1, requires_grad=True)
        target = torch.tensor([[[[1, 0], [1, 0]]]])
        valid = torch.tensor([[[[True, True], [True, False]]]])
        actual = gaussian_dense_dice_loss(logits, target, valid)
        expected = 1 - (2 * .5 * (2 / 3) * .75 + 1) / ((.5 + 2 / 3) * .75 + 1)
        self.assertAlmostEqual(actual.item(), expected, places=6)
        actual.backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        empty = gaussian_dense_dice_loss(logits, target, torch.zeros_like(valid))
        self.assertEqual(empty.item(), 0)

    def test_invalid_content_scale_rejected(self):
        for value in (0., -1., math.inf, math.nan):
            with self.assertRaises(ValueError):
                GaussianPrimitivePredictorV81((8, 16, 32, 64), channels=16, content_scale=value)


class V81ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        register_all_modules(init_default_scope=True)

    def setUp(self):
        torch.manual_seed(42)

    def test_full_loss_has_no_dead_parameters(self):
        model = tiny_model()
        images, samples = batch()
        losses = model.loss(images, samples)
        self.assertEqual(set(losses), {'loss_final', 'loss_coarse', 'loss_sdf', 'loss_boundary_band',
                                      'loss_region_consistency', 'loss_final_boundary',
                                      'loss_gaussian_dense', 'loss_gaussian_node', 'loss_gaussian_dense_dice'})
        objective(losses).backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_final_loss_trains_new_modules_and_gaussian_parameters(self):
        model = tiny_model()
        model.loss(*batch())['loss_final'].backward()
        groups = ('gaussian_predictor.content_query', 'gaussian_predictor.content_key',
                  'gaussian_predictor.center_head', 'gaussian_predictor.covariance_head',
                  'gaussian_predictor.amplitude_head', 'gaussian_predictor.height_head',
                  'gaussian_graph.layers.0.edge_value', 'gaussian_graph.layers.1.edge_value',
                  'local_fusion.0.rgb_projection', 'local_fusion.1.rgb_projection')
        for group in groups:
            gradients = [p.grad for name, p in model.named_parameters() if name.startswith(group)]
            self.assertTrue(gradients, group)
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), group)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, group)

    def test_content_only_final_loss_trains_content_and_retained_geometry(self):
        model = tiny_model(gaussian_cfg=dict(grid_sizes=(2,), max_size=16, spatial_support=False))
        model.loss(*batch())['loss_final'].backward()
        for head in ('content_query', 'content_key', 'center_head', 'covariance_head',
                     'amplitude_head', 'height_head'):
            parameters = list(getattr(model.gaussian_predictor, head).parameters())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), head)
            self.assertGreater(sum(p.grad.abs().sum().item() for p in parameters), 0, head)

    def test_no_gaussian_aux_has_no_heads_or_auxiliary_losses(self):
        model = tiny_model(gaussian_dense_weight=0, gaussian_node_weight=0, gaussian_dense_dice_weight=0)
        self.assertFalse(hasattr(model, 'gaussian_dense_head'))
        self.assertFalse(hasattr(model, 'gaussian_node_head'))
        losses = model.loss(*batch())
        self.assertFalse(any(name.startswith('loss_gaussian') for name in losses))
        self.assertIn('loss_region_consistency', losses)
        self.assertIn('loss_final_boundary', losses)
        objective(losses).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.gaussian_predictor.content_query.weight.grad.abs().sum().item(), 0)

    def test_all_new_switches_off_reproduce_v8(self):
        before = v8_model().eval()
        after = tiny_model(gaussian_cfg=dict(grid_sizes=(2,), max_size=16, content_aware=False),
                           graph_cfg=dict(heads=4, layers=2, use_edge_value=False),
                           local_detail=False, region_loss_weight=0, final_boundary_loss_weight=0,
                           balanced_boundary=False, gaussian_dense_dice_weight=0).eval()
        after.load_state_dict(before.state_dict(), strict=True)
        images, samples = batch()
        torch.testing.assert_close(before.encode_decode(images), after.encode_decode(images))
        left, right = before.loss(images, samples), after.loss(images, samples)
        self.assertEqual(set(left), set(right))
        for name in left:
            torch.testing.assert_close(left[name], right[name])

    def test_all_ignore_is_connected_zero(self):
        for classes in (None, 3):
            with self.subTest(classes=classes):
                options = {} if classes is None else dict(type='RPGVRemoteSensingV81', num_classes=classes)
                model = tiny_model(**options)
                images, samples = batch(label=255)
                losses = model.loss(images, samples)
                for name, loss in losses.items():
                    self.assertEqual(loss.item(), 0, name)
                    self.assertTrue(loss.requires_grad, name)
                objective(losses).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters() if p.requires_grad))

    def test_each_switch_keeps_forward_backward_valid(self):
        settings = [dict(use_gaussians=False), dict(use_message_passing=False),
                    dict(gaussian_cfg=dict(grid_sizes=(2,), max_size=16, fixed_gaussians=True)),
                    dict(gaussian_cfg=dict(grid_sizes=(2, 4), max_size=16)),
                    dict(gaussian_cfg=dict(grid_sizes=(2,), max_size=16, spatial_support=False)),
                    dict(gaussian_cfg=dict(grid_sizes=(2,), max_size=16, content_aware=False)),
                    dict(graph_cfg=dict(heads=4, layers=2, use_edge_value=False)),
                    dict(local_detail=False), dict(region_loss_weight=0, final_boundary_loss_weight=0),
                    dict(balanced_boundary=False), dict(gaussian_dense_dice_weight=0),
                    dict(gaussian_dense_weight=0, gaussian_node_weight=0, gaussian_dense_dice_weight=0),
                    dict(gaussian_dense_weight=0, gaussian_node_weight=0)]
        for options in settings:
            with self.subTest(options=options):
                model = tiny_model(**options)
                objective(model.loss(*batch())).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if options.get('use_gaussians') is False:
                    self.assertFalse(hasattr(model, 'gaussian_predictor'))
                if options.get('use_message_passing') is False:
                    self.assertFalse(hasattr(model, 'gaussian_graph'))
                if options.get('gaussian_dense_dice_weight') == 0:
                    self.assertNotIn('loss_gaussian_dense_dice', model.loss(*batch()))

    def test_public_classes_predict_and_slide(self):
        for classes in (2, 7, 8):
            with self.subTest(classes=classes):
                model = tiny_model(type='RPGVRemoteSensingV81', num_classes=classes).eval()
                images, samples = batch(size=(64, 80))
                for sample in samples:
                    sample.gt_sem_seg.data = torch.randint(0, classes, sample.gt_sem_seg.data.shape)
                losses = model.loss(images, samples)
                self.assertNotIn('loss_gaussian_dense_dice', losses)
                objective(losses).backward()
                with torch.no_grad():
                    self.assertEqual(model.encode_decode(images).shape, (2, classes, 64, 80))
                    model.test_cfg = dict(mode='slide', crop_size=(64, 64), stride=(48, 48))
                    prediction = model.predict(images, samples)
                    self.assertEqual(prediction[0].pred_sem_seg.data.shape, (1, 64, 80))
                    self.assertTrue(torch.isfinite(prediction[0].seg_logits.data).all())

    def test_checkpoint_roundtrip_and_rgb_contract(self):
        model = tiny_model().eval()
        restored = tiny_model().eval()
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        images, _ = batch()
        with torch.no_grad():
            torch.testing.assert_close(model.encode_decode(images), restored.encode_decode(images))
        with self.assertRaisesRegex(ValueError, 'three BGR'):
            model.encode_decode(torch.rand(2, 5, 64, 64))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_autocast_and_optimizer_update(self):
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            with self.subTest(dtype=dtype):
                model = tiny_model().cuda().train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
                images, samples = batch()
                initial = model.gaussian_graph.layers[0].edge_value.weight.detach().clone()
                with torch.autocast('cuda', dtype=dtype):
                    loss = objective(model.loss(images.cuda(), samples))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                optimizer.step()
                self.assertFalse(torch.equal(initial, model.gaussian_graph.layers[0].edge_value.weight))


if __name__ == '__main__':
    unittest.main()
