"""V8.1 configuration and orchestration contracts (no model construction)."""
import contextlib
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mmengine.config import Config

from scripts import run_experiments as runner
from wwtpseg.experiments import V81_ABLATIONS


class V81IntegrationTest(unittest.TestCase):
    ABLATIONS = {
        'no_edge_value': {'graph_cfg': {'use_edge_value': False}},
        'no_content_support': {'gaussian_cfg': {'content_aware': False}},
        'no_local_detail': {'local_detail': False},
        'no_pairwise': {'region_loss_weight': 0.0, 'final_boundary_loss_weight': 0.0},
        'no_dense_dice': {'gaussian_dense_dice_weight': 0.0},
        'unbalanced_band': {'balanced_boundary': False},
    }

    def load(self, dataset, name='rpgv_v81'):
        return Config.fromfile(runner.experiment_config(dataset, name))

    def test_roster_and_paths(self):
        expected = tuple('rpgv_v81_' + suffix for suffix in self.ABLATIONS)
        self.assertEqual(V81_ABLATIONS, expected)
        self.assertEqual(runner.EXPERIMENT_SUITES['v81'], ('rpgv_v81',))
        self.assertEqual(runner.EXPERIMENT_SUITES['v81-ablations'], expected)
        self.assertEqual(runner.EXPERIMENT_SUITES['v81-all'],
                         runner.MAINSTREAM_MODELS + ('rpgv_v81',) + expected)
        for name in ('rpgv_v81',) + expected:
            self.assertIn(name, runner.OPTIONAL_MODELS)
            self.assertNotIn(name, runner.OFFLINE_GEOMETRY_MODELS)
            self.assertTrue(runner.experiment_config('wwtp', name).is_file())
            for suite, roster in runner.EXPERIMENT_SUITES.items():
                if not suite.startswith('v81'):
                    self.assertNotIn(name, roster)
        for dataset in ('loveda', 'uavid', 'deepglobe'):
            for name in expected:
                with self.assertRaisesRegex(ValueError, 'only available for WWTP'):
                    runner.experiment_config(dataset, name)

    def test_defaults_and_inherited_rgb_protocol(self):
        for dataset in runner.DATASETS:
            with self.subTest(dataset=dataset):
                cfg = self.load(dataset)
                reference = self.load(dataset, 'rpgv_v8')
                model = cfg.model
                self.assertEqual(model.type, 'RPGVNetV81' if dataset == 'wwtp'
                                 else 'RPGVRemoteSensingV81')
                self.assertEqual(model.gaussian_cfg.grid_sizes, (4, 8, 16))
                self.assertEqual(model.gaussian_cfg.max_size, 64)
                self.assertTrue(model.gaussian_cfg.content_aware)
                self.assertEqual(model.gaussian_cfg.content_scale, 2.0)
                self.assertTrue(model.graph_cfg.use_edge_value)
                self.assertTrue(model.local_detail)
                self.assertTrue(model.balanced_boundary)
                self.assertEqual(model.region_loss_weight, 0.05)
                self.assertEqual(model.final_boundary_loss_weight, 0.1)
                self.assertEqual(model.gaussian_dense_dice_weight, 0.05)
                self.assertEqual(model.gaussian_dense_weight, 0.1)
                self.assertEqual(model.gaussian_node_weight, 0.05)
                self.assertEqual(model.rgb_encoder.in_channels, 3)
                self.assertEqual(model.test_cfg, reference.model.test_cfg)
                if dataset != 'wwtp':
                    self.assertEqual(model.num_classes, {'loveda': 7, 'uavid': 8,
                                                        'deepglobe': 2}[dataset])
                # Everything except model and output directory is the unmodified V8
                # protocol: no claimed batch measurement or new precision override.
                actual, previous = cfg.to_dict(), reference.to_dict()
                for key in ('model', 'work_dir'):
                    actual.pop(key)
                    previous.pop(key)
                self.assertEqual(actual, previous)
                self.assertEqual(cfg.work_dir, f'work_dirs/{dataset}/rpgv_v81')
                self.assertNotIn('protocol_amp_override', cfg)
                accum = cfg.optim_wrapper.accumulative_counts
                self.assertEqual(cfg.train_dataloader.batch_size * accum, 8)
                self.assertEqual(cfg.train_cfg.max_iters, 7500 * accum)
                self.assertEqual(cfg.train_cfg.val_interval, 250 * accum)
                self.assertEqual(cfg.default_hooks.checkpoint.interval, 250 * accum)
                self.assertEqual(cfg.optim_wrapper.dtype, 'bfloat16')
                for split in ('train', 'val', 'test'):
                    steps = [s.type for s in cfg[f'{split}_dataloader'].dataset.pipeline]
                    self.assertNotIn('LoadPseudoGeometry', steps)
                    self.assertEqual(steps[-1], 'PackSegInputs')

    def test_ablations_change_only_the_requested_objective_or_module(self):
        full = self.load('wwtp').to_dict()
        for suffix, changes in self.ABLATIONS.items():
            name = 'rpgv_v81_' + suffix
            with self.subTest(model=name):
                expected = copy.deepcopy(full)
                for key, value in changes.items():
                    if isinstance(value, dict):
                        expected['model'][key].update(value)
                    else:
                        expected['model'][key] = value
                expected['work_dir'] = f'work_dirs/wwtp/{name}'
                self.assertEqual(self.load('wwtp', name).to_dict(), expected)

    def test_dependency_free_dry_runs(self):
        selections = [(dataset, 'v81') for dataset in runner.DATASETS]
        selections += [('wwtp', suite) for suite in ('v81-ablations', 'v81-all')]
        with tempfile.TemporaryDirectory() as directory:
            for dataset, suite in selections:
                with self.subTest(dataset=dataset, suite=suite):
                    result = subprocess.run([
                        sys.executable, '-S', str(runner.ROOT / 'scripts/run_experiments.py'),
                        'train-test', '--dataset', dataset, '--suite', suite,
                        '--work-root', directory, '--dry-run'],
                        cwd=directory, capture_output=True, text=True, check=True)
                    count = len(runner.EXPERIMENT_SUITES[suite])
                    self.assertEqual(result.stdout.count('tools/train.py'), count)
                    self.assertEqual(result.stdout.count('tools/test.py'), count)
                    self.assertNotIn('generate_pseudo_geometry', result.stdout)
                    self.assertNotIn('[PIPELINE-AUTO]', result.stdout)
                    self.assertNotIn('[GATE]', result.stdout)
                    split = runner.EVALUATION_PROTOCOLS[dataset]['split']
                    self.assertIn(f'--split {split}', result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_real_ablation_planning_has_no_geometry_gate_or_retraining(self):
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()), \
                patch.object(runner, 'prepare_geometry') as prepare, \
                patch.object(runner, 'check_v7_superiority') as gate, \
                patch.object(runner, 'best_checkpoint', return_value=Path('/tmp/best.pth')), \
                patch.object(runner, 'run_command') as execute:
            runner.main(['train-test', '--suite', 'v81-ablations', '--work-root', directory,
                         '--gate-v7-superiority', '--auto-prerequisites'])
            prepare.assert_not_called()
            gate.assert_not_called()
            self.assertEqual(execute.call_count, 2 * len(V81_ABLATIONS))
            for call in execute.call_args_list:
                command = call.args[0]
                self.assertIn(command[1], ('tools/train.py', 'tools/test.py'))
                self.assertIn('configs/rpgv/v81/', command[2])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_mixed_selection_prepares_only_legacy_geometry(self):
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()), \
                patch.object(runner, 'prepare_geometry') as prepare, \
                patch.object(runner, 'run_command') as execute:
            runner.main(['train', '--models', 'rpgv_v7', 'rpgv_v81',
                         '--work-root', directory])
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.args[0],
                             [runner.experiment_config('wwtp', 'rpgv_v7')])
            self.assertEqual(execute.call_count, 2)


if __name__ == '__main__':
    unittest.main()
