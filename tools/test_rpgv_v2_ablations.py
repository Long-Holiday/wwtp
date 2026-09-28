#!/usr/bin/env python3
"""Regression checks for ablation isolation, task plans and result selection."""
import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace

import numpy as np
import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_rpgv_v2_ablations import (BUDGET, CORE, PROGRESSIVE, RUN, VARIANTS, atomic_json,
                                     make_config, prepare, run_experiment, select_best,
                                     parse_args, file_hash, evaluate, training_result)
from wwtpseg.engine import RPGVEarlyStoppingHook
from summarize_rpgv_v2_ablations import collect, aggregate
from analyze_wwtp_topology import topology
from smoke_test import _sample, _add_pseudo_geometry
from wwtpseg.evaluation.binary_topology_metric import BinaryTopologyMetric, mask_topology
from wwtpseg.models.utils.rpgv_v2_modules import UnifiedContourHead


def validation(work_dir, rows):
    path = work_dir / '20260922/vis_data/scalars.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(dict(step=s, **{'binary/Foreground_IoU': v})) + '\n'
                            for s, v in rows))


class AblationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        torch.set_num_threads(2)

    def test_defaults_and_reject_stage_reuse(self):
        with patch.object(sys, 'argv', ['runner', 'plan']):
            args = parse_args()
        self.assertEqual(args.variants, list(PROGRESSIVE))
        self.assertEqual(args.protocol, 'single_stage')
        self.assertEqual(args.max_iters, BUDGET)
        for options in (['--protocol', 'paper_reuse'], ['--stage2-checkpoint', 'old.pth'],
                        ['--max-iters', '1500']):
            with patch.object(sys, 'argv', ['runner', 'plan', *options]), self.assertRaises(SystemExit):
                parse_args()

    def test_configs_and_progressive_chain(self):
        for variant in VARIANTS:
            cfg = make_config(variant, 42, Path('/tmp/unused'), max_iters=4000)
            self.assertEqual(cfg.model.training_stage, 'joint')
            self.assertFalse(cfg.model.learnable_loss_weights)
            self.assertIsNone(cfg.load_from)
            self.assertNotIn('required_previous_stage', cfg)
            self.assertTrue(cfg.enable_early_stopping)
            self.assertEqual(cfg.train_cfg.max_iters, cfg.param_scheduler[-1].end)
            self.assertEqual(cfg.train_cfg.max_iters, 4000)
            self.assertNotIn('decoder', cfg.optim_wrapper.paramwise_cfg.custom_keys)
            self.assertNotIn('loss_log_variances', cfg.optim_wrapper.paramwise_cfg.custom_keys)
            hook = cfg.custom_hooks[0]
            self.assertEqual((hook.type, hook.min_delta, hook.patience),
                             ('RPGVEarlyStoppingHook', 0.1, 5))
        for i, variant in enumerate(PROGRESSIVE):
            cfg = make_config(variant, 42, Path('/tmp/unused'))
            c = cfg.model.component_cfg
            self.assertEqual(c.depth_rectification, i >= 1)
            self.assertEqual(c.learned_reliability, i >= 1)
            self.assertEqual(c.frequency_validation, i >= 2)
            self.assertEqual(c.reliability_weighting, i >= 3)
            self.assertEqual(cfg.model.use_contour, i >= 4)
            self.assertEqual(cfg.model.region_loss_weight > 0, i >= 5)

    def test_joint_gradients_and_fixed_weights_all_variants(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = make_config(variant, 42, Path('/tmp/unused'))
                cfg.model.rgb_encoder.init_cfg = None
                cfg.model.global_thumbnail_size = 64
                cfg.model.geometry_dropout_prob = 0
                model = MODELS.build(cfg.model).train()
                image, sample = _sample(64)
                image = _add_pseudo_geometry(image).unsqueeze(0)
                losses = model.loss(image, [sample])
                total, _ = model.parse_losses(losses)
                torch.testing.assert_close(total, sum(v for k, v in losses.items() if 'loss' in k))
                total.backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                before = model.refiner.coarse_head.weight.detach().clone()
                weights_before = dict(model.loss_weights)
                self.assertFalse(hasattr(model, 'loss_log_variances'))
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                optimizer.step()
                self.assertFalse(torch.equal(before, model.refiner.coarse_head.weight))
                self.assertEqual(model.loss_weights, weights_before)
                self.assertEqual(model.loss_weights['rgb'], model.loss_weights['geometry'])
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                self.assertTrue(any(p.requires_grad for p in model.rgb_encoder.parameters()))
                self.assertTrue(any(p.requires_grad for p in model.geometry_encoder.parameters()))
                if not model.component_enabled('depth_rectification'):
                    for term in ('loss_preserve', 'loss_equivariance', 'loss_correction_smoothness'):
                        self.assertNotIn(term, losses)
                if model.region_loss_weight == 0:
                    self.assertNotIn('loss_region_consistency', losses)
                    self.assertNotIn('loss_final_boundary', losses)
                restored = MODELS.build(cfg.model)
                restored.load_state_dict(model.state_dict(), strict=True)
                torch.testing.assert_close(restored.refiner.coarse_head.weight,
                                           model.refiner.coarse_head.weight)

    def test_early_stop_threshold_patience_resume_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config('full', 42, Path(tmp))
            options = dict(cfg.custom_hooks[0])
            options.pop('type')
            hook = RPGVEarlyStoppingHook(**options)
            runner = SimpleNamespace(iter=0, work_dir=tmp, logger=Mock(),
                                     train_loop=SimpleNamespace(stop_training=False))
            scores = [(2000, 80), (4000, 80.04), (6000, 80.08),
                      (8000, 80.02), (10000, 79.9), (12000, 80.09)]
            for step, score in scores[:3]:
                hook.after_val_epoch(runner, {'binary/Foreground_IoU': score})
            self.assertEqual(hook.wait_count, 2)
            validation(Path(tmp), scores)  # Later logs must not leak into resume.
            runner.iter = 6000
            resumed = RPGVEarlyStoppingHook(**options)
            resumed.before_train(runner)
            self.assertEqual(resumed.wait_count, 2)
            for step, score in scores[3:]:
                resumed.after_val_epoch(runner, {'binary/Foreground_IoU': score})
                self.assertEqual(runner.train_loop.stop_training, step == 12000)
            improved = RPGVEarlyStoppingHook(**options)
            for score in (80, 80.04, 80.11):
                improved.after_val_epoch(runner, {'binary/Foreground_IoU': score})
            self.assertEqual(improved.wait_count, 0)
            with self.assertRaises(FloatingPointError):
                resumed.after_val_epoch(runner, {'binary/Foreground_IoU': float('nan')})

    def test_early_stopped_run_selection_skip_evaluation_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, spec = prepare(tmp, 'single_stage', 'full', 42, 20000)
            work = directory / RUN
            scores = [(2000, 80), (4000, 80.04), (6000, 80.08),
                      (8000, 80.02), (10000, 79.9), (12000, 80.09)]
            def fake_train(command, **kwargs):
                validation(work, scores)
                (work / 'best_binary_Foreground_IoU_iter_12000.pth').write_bytes(b'best')
            with patch('run_rpgv_v2_ablations.subprocess.run', side_effect=fake_train) as train:
                run_experiment(directory, spec)
                run_experiment(directory, spec)
                self.assertEqual(train.call_count, 1)
            done = json.loads((work / 'completed.json').read_text())
            self.assertTrue(done['stopped_early'])
            self.assertEqual(done['last_val_iter'], 12000)
            self.assertEqual(done['validation']['selected_iter'], 12000)
            row = collect(Path(tmp), 'val')[0]
            self.assertEqual(row['status'], 'complete')
            self.assertTrue(row['stopped_early'])
            self.assertEqual(row['loss_weighting'], 'fixed')
            with patch('run_rpgv_v2_ablations.subprocess.run') as evaluate_call:
                evaluate(directory, 'val')
                self.assertEqual(evaluate_call.call_count, 1)
            validation(work, scores[:3])
            with self.assertRaises(RuntimeError):
                training_result(work, 20000, spec['early_stopping'])
            self.assertEqual(collect(Path(tmp), 'val')[0]['status'], 'incomplete')

    def test_selection_requires_budget_and_first_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            validation(work, [(2000, 80)])
            with self.assertRaises(RuntimeError):
                select_best(work, 4000)
            validation(work, [(2000, 80), (4000, 80)])
            checkpoint = work / 'best_binary_Foreground_IoU_iter_2000.pth'
            checkpoint.touch()
            best, metrics = select_best(work, 4000)
            self.assertEqual(best, checkpoint)
            self.assertEqual(metrics['selected_iter'], 2000)

    def test_independent_run_idempotence_and_config_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, identity = prepare(tmp, 'single_stage', 'full', 42, 4000)
            self.assertEqual(prepare(tmp, 'single_stage', 'full', 42, 4000)[1], identity)
            calls = []
            def fake_train(command, **kwargs):
                calls.append(command)
                work = Path(command[command.index('--work-dir') + 1])
                validation(work, [(4000, 80)])
                (work / 'best_binary_Foreground_IoU_iter_4000.pth').write_bytes(b'checkpoint')
            with patch('run_rpgv_v2_ablations.subprocess.run', side_effect=fake_train):
                run_experiment(directory, identity)
                run_experiment(directory, identity)
            self.assertEqual(len(calls), 1)
            self.assertNotIn('--cfg-options', calls[0])
            self.assertNotIn('--resume', calls[0])
            (directory / 'configs/joint.py').write_text('load_from = "stage2.pth"')
            with self.assertRaises(RuntimeError):
                run_experiment(directory, identity)
            with self.assertRaises(RuntimeError):
                prepare(tmp, 'single_stage', 'full', 42, 4000)

    def test_resume_only_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, spec = prepare(tmp, 'single_stage', 'full', 42, 4000)
            work = directory / RUN
            work.mkdir()
            atomic_json(work / 'started.json', dict(initialization=None))
            external = Path(tmp) / 'stage2.pth'
            external.write_bytes(b'old')
            (work / 'last_checkpoint').write_text(str(external))
            with self.assertRaises(RuntimeError):
                run_experiment(directory, spec, resume=True)
            local = work / 'iter_2000.pth'
            local.write_bytes(b'local')
            (work / 'last_checkpoint').write_text(str(local))
            with self.assertRaises(RuntimeError):
                run_experiment(directory, spec)
            def fake_train(command, **kwargs):
                self.assertIn('--resume', command)
                validation(work, [(4000, 80)])
                (work / 'best_binary_Foreground_IoU_iter_4000.pth').write_bytes(b'best')
            with patch('run_rpgv_v2_ablations.subprocess.run', side_effect=fake_train):
                run_experiment(directory, spec, resume=True)

    def test_incomplete_excluded_and_paired_progressive_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed, variant, score, complete in [(42, 'full', 80, True),
                                                   (42, 'progressive_base', 76, True),
                                                   (42, 'progressive_rgr', 78, True),
                                                   (7, 'progressive_rgr', 99, False)]:
                directory, _ = prepare(tmp, 'single_stage', variant, seed, 4000)
                work = directory / RUN
                validation(work, [(4000 if complete else 2000, score)])
                if complete:
                    atomic_json(work / 'completed.json', dict(budget=4000, last_val_iter=4000,
                        checkpoint_sha256='abc', validation={'selected_iter': 4000,
                                                           'binary/Foreground_IoU': score}))
            rows = collect(Path(tmp), 'test')
            row = next(r for r in rows if r['variant'] == 'progressive_rgr' and r['seed'] == 42)
            self.assertEqual(row['delta_val_iou'], -2)
            self.assertEqual(row['delta_previous_val_iou'], 2)
            self.assertIsNone(row['iou'])
            group = next(g for g in aggregate(rows) if g['variant'] == 'progressive_rgr')
            self.assertEqual(group['val_iou']['n'], 1)

    def test_topology_matches_offline_audit_and_ignore(self):
        gt = np.zeros((32, 32), dtype=np.uint8)
        gt[8:24, 8:24] = 1
        pred = gt.copy()
        pred[12:14, 12:14] = 0
        pred[2:4, 2:4] = 1
        valid = gt != 255
        self.assertEqual(mask_topology(pred.astype(bool), valid, 256),
                         topology(pred.astype(bool), valid, 256))
        metric = BinaryTopologyMetric()
        metric.process({}, [dict(gt_sem_seg=dict(data=gt), pred_sem_seg=dict(data=pred))])
        result = metric.compute_metrics(metric.results)
        self.assertEqual(result['pred_holes'], 1)
        self.assertEqual(result['pred_hole_pixels'], 4)
        self.assertEqual(result['detached_fp_pixels'], 4)
        self.assertEqual(result['detached_fp_components'], 1)
        valid[12:14, 12:14] = False
        self.assertEqual(mask_topology(pred.astype(bool), valid, 256)['holes'], 0)

    def test_cpu_evaluator_selects_val_and_caches_result(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_config('full', 42, root / 'train')
            cfg.model.rgb_encoder.init_cfg = None
            cfg.model.global_thumbnail_size = 64
            cfg.model.data_preprocessor.size = (64, 64)
            cfg.model.test_cfg = dict(mode='whole')
            for split in ('val', 'test'):
                for folder in ('images', 'annotations', 'pseudo_geometry'):
                    (root / folder / split).mkdir(parents=True)
                Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(root / 'images' / split / 'case.png')
                # Test has no foreground, so accidentally using it for val is detectable.
                label = np.zeros((64, 64), dtype=np.uint8)
                if split == 'val':
                    label[16:48, 16:48] = 1
                Image.fromarray(label).save(root / 'annotations' / split / 'case.png')
                np.savez(root / 'pseudo_geometry' / split / 'case.npz',
                         depth=np.full((64, 64), 30000, dtype=np.uint16),
                         reliability=np.full((64, 64), 255, dtype=np.uint8))
                loader = cfg[f'{split}_dataloader']
                loader.num_workers = 0
                loader.persistent_workers = False
                loader.dataset.data_root = str(root)
                for transform in loader.dataset.pipeline:
                    if transform.type == 'LoadPseudoGeometry':
                        transform.pseudo_root = str(root / 'pseudo_geometry')
                    elif transform.type == 'GenerateGlobalThumbnail':
                        transform.size = (64, 64)
            config = root / 'config.py'
            cfg.dump(config)
            checkpoint = root / 'model.pth'
            torch.save(dict(state_dict=MODELS.build(cfg.model).state_dict()), checkpoint)
            command = [sys.executable, 'tools/evaluate_rpgv_v2_ablation.py',
                       str(config), str(checkpoint), '--split', 'val', '--output', str(root / 'eval')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = json.loads((root / 'eval/metrics.json').read_text())
            self.assertEqual(output['split'], 'val')
            self.assertEqual(output['metrics']['topology/images'], 1)
            self.assertEqual(output['metrics']['topology/gt_components'], 1)
            self.assertEqual(output['metrics']['topology/negative_images'], 0)
            cached = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(cached.returncode, 0, cached.stderr)
            self.assertIn('[SKIP evaluated]', cached.stdout)


if __name__ == '__main__':
    unittest.main()
