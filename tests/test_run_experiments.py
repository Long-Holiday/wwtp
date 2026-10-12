"""Dependency-free regression tests for experiment orchestration."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from scripts import run_experiments as runner


class ExperimentRunnerTest(unittest.TestCase):
    def run_main(self, args, side_effect=None):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(
                runner.subprocess, 'run', side_effect=side_effect) as execute:
            runner.main(args)
        return execute

    def test_roster_and_configs(self):
        self.assertEqual({key: len(value) for key, value in runner.EXPERIMENT_SUITES.items()},
                         {'baselines': 7, 'v6': 1, 'v6-ablations': 5, 'all': 13,
                          'v7': 1, 'v7-ablations': 6, 'v7-all': 15,
                          'v8': 1, 'v8-ablations': 7, 'v8-all': 15})
        self.assertEqual(len(set(runner.EXPERIMENT_SUITES['all'])), 13)
        for name in runner.EXPERIMENT_SUITES['all']:
            self.assertTrue(runner.experiment_config('wwtp', name).is_file())
        for dataset in ('uavid', 'loveda'):
            self.assertTrue(runner.experiment_config(dataset, 'rpgv_v6').is_file())
            with self.assertRaises(ValueError):
                runner.experiment_config(dataset, 'rpgv_v6_rgb_only')

    def test_dependency_free_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            script = runner.ROOT / 'scripts/run_experiments.py'
            result = subprocess.run([sys.executable, '-S', str(script), 'train-test',
                                     '--suite', 'all', '--work-root', directory, '--dry-run'],
                                    cwd=directory, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.count('tools/train.py'), 13)
            self.assertEqual(result.stdout.count('tools/test.py'), 13)
            self.assertEqual(result.stdout.count('tools/data/generate_pseudo_geometry.py'), 1)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_v7_roster_and_dependency_free_dry_run(self):
        for name in runner.EXPERIMENT_SUITES['v7-all']:
            self.assertTrue(runner.experiment_config('wwtp', name).is_file())
        for dataset in ('loveda', 'uavid'):
            self.assertTrue(runner.experiment_config(dataset, 'rpgv_v7').is_file())
            with self.assertRaises(ValueError):
                runner.experiment_config(dataset, 'rpgv_v7_fixed_grid')
        result = subprocess.run([sys.executable, '-S', str(runner.ROOT / 'scripts/run_experiments.py'),
                                 'train-test', '--suite', 'v7-all', '--dry-run'],
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.count('tools/train.py'), 15)
        self.assertEqual(result.stdout.count('tools/test.py'), 15)
        self.assertEqual(result.stdout.count('tools/data/generate_pseudo_geometry.py'), 1)
        self.assertIn('configs/rpgv/v7/rpgv_v7.py', result.stdout)
        self.assertNotIn('rpgv_v7', runner.EXPERIMENT_SUITES['all'])

    def test_v7_core_analysis_and_legacy_selections(self):
        expected = tuple('rpgv_v7_' + suffix for suffix in (
            'rgb_only', 'fixed_grid', 'no_feedback', 'no_relation_supervision',
            'no_composition', 'no_structure_aux'))
        analysis = tuple('rpgv_v7_' + suffix for suffix in (
            'no_hierarchical_supervision', 'no_node_supervision', 'no_message_passing'))
        legacy = tuple('rpgv_v7_' + suffix for suffix in (
            'no_contrastive', 'no_relational_supervision', 'no_edges',
            'no_perturbation', 'rgb_capacity'))
        self.assertEqual(runner.EXPERIMENT_SUITES['v7-ablations'], expected)
        self.assertEqual(len(set(runner.OPTIONAL_MODELS)), len(runner.OPTIONAL_MODELS))
        for name in expected + analysis + legacy:
            self.assertIn(name, runner.OPTIONAL_MODELS)
            self.assertTrue(runner.experiment_config('wwtp', name).is_file())
            for dataset in ('uavid', 'loveda', 'deepglobe'):
                with self.assertRaises(ValueError):
                    runner.experiment_config(dataset, name)
        for name in analysis + legacy:
            self.assertNotIn(name, runner.EXPERIMENT_SUITES['v7-all'])
        for selection, names in ((['--suite', 'v7-ablations'], expected),
                                 (['--models', *analysis], analysis),
                                 (['--models', *legacy], legacy)):
            result = subprocess.run([
                sys.executable, '-S', str(runner.ROOT / 'scripts/run_experiments.py'),
                'train-test', *selection, '--dry-run'],
                capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.count('tools/train.py'), len(names))
            self.assertEqual(result.stdout.count('tools/test.py'), len(names))
            for name in names:
                config = str(runner.experiment_config('wwtp', name).relative_to(runner.ROOT))
                self.assertIn(config, result.stdout)
                self.assertIn(f'work_dirs/wwtp/{name}', result.stdout)
            if selection[0] == '--suite':
                for name in analysis + legacy:
                    config = str(runner.experiment_config('wwtp', name).relative_to(runner.ROOT))
                    self.assertNotIn(config, result.stdout)
                    self.assertNotIn(f'work_dirs/wwtp/{name}', result.stdout)

    def test_baseline_variant_selection(self):
        self.assertIn('mask2former_q10', runner.OPTIONAL_MODELS)
        self.assertTrue(runner.experiment_config('wwtp', 'mask2former_q10').is_file())
        for dataset in ('uavid', 'loveda', 'deepglobe'):
            with self.assertRaises(ValueError):
                runner.experiment_config(dataset, 'mask2former_q10')
        result = subprocess.run([sys.executable, '-S', str(runner.ROOT / 'scripts/run_experiments.py'),
                                 'train-test', '--models', 'mask2former_q10', '--dry-run'],
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.count('tools/train.py'), 1)
        self.assertEqual(result.stdout.count('tools/test.py'), 1)
        self.assertIn('configs/wwtp/mask2former_q10.py', result.stdout)
        self.assertIn('work_dirs/wwtp/mask2former_q10', result.stdout)

    def test_exclude_drops_models_from_a_suite(self):
        script = str(runner.ROOT / 'scripts/run_experiments.py')
        full = subprocess.run([sys.executable, '-S', script, 'train-test', '--suite', 'baselines',
                               '--dry-run'], capture_output=True, text=True, check=True)
        filtered = subprocess.run([sys.executable, '-S', script, 'train-test', '--suite', 'baselines',
                                   '--exclude', 'mask2former', 'hdnet', '--dry-run'],
                                  capture_output=True, text=True, check=True)
        self.assertEqual(full.stdout.count('tools/train.py'), len(runner.EXPERIMENT_SUITES['baselines']))
        self.assertEqual(filtered.stdout.count('tools/train.py'), full.stdout.count('tools/train.py') - 2)
        self.assertNotIn('configs/wwtp/mask2former.py', filtered.stdout)
        self.assertNotIn('configs/wwtp/hdnet.py', filtered.stdout)
        self.assertIn('configs/wwtp/mask2former.py', full.stdout)
        self.assertIn('configs/wwtp/hdnet.py', full.stdout)

    def test_v8_dry_runs_need_no_geometry_dependencies(self):
        script = str(runner.ROOT / 'scripts/run_experiments.py')
        for dataset in runner.DATASETS:
            self.assertTrue(runner.experiment_config(dataset, 'rpgv_v8').is_file())
            result = subprocess.run([sys.executable, '-S', script, 'train-test',
                                     '--dataset', dataset, '--suite', 'v8', '--dry-run'],
                                    capture_output=True, text=True, check=True)
            self.assertNotIn('generate_pseudo_geometry', result.stdout)
            self.assertEqual(result.stdout.count('tools/train.py'), 1)
            self.assertEqual(result.stdout.count('tools/test.py'), 1)
        for suite in ('v8-ablations', 'v8-all'):
            result = subprocess.run([sys.executable, '-S', script, 'train-test',
                                     '--suite', suite, '--dry-run'],
                                    capture_output=True, text=True, check=True)
            self.assertNotIn('generate_pseudo_geometry', result.stdout)
            self.assertEqual(result.stdout.count('tools/train.py'), len(runner.EXPERIMENT_SUITES[suite]))
        self.assertNotIn('rpgv_v8', runner.OFFLINE_GEOMETRY_MODELS)
        self.assertNotIn('rpgv_v8', runner.EXPERIMENT_SUITES['v7-all'])

    def test_v8_real_planning_skips_depth_and_mixed_selection_is_precise(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'prepare_geometry') as prepare:
            calls = self.run_main(['train', '--suite', 'v8', '--work-root', directory]).call_args_list
            self.assertEqual(len(calls), 1)
            prepare.assert_not_called()
            self.run_main(['train', '--models', 'rpgv_v7', 'rpgv_v8', '--work-root', directory])
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.args[0], [runner.experiment_config('wwtp', 'rpgv_v7')])

    def test_train_then_validation_selected_test(self):
        with tempfile.TemporaryDirectory() as directory:
            def train(command, **kwargs):
                if 'tools/train.py' in command:
                    output = Path(command[command.index('--work-dir') + 1])
                    output.mkdir(parents=True)
                    (output / 'best_binary_Foreground_IoU_iter_1000.pth').touch()
                    (output / 'iter_20000.pth').touch()
            calls = self.run_main(['train-test', '--models', 'hrnet', 'hdnet',
                                   '--work-root', directory, '--no-pretrained',
                                   '--cfg-options', 'randomness.seed=123'], train).call_args_list
            self.assertEqual(len(calls), 4)
            for index, call in enumerate(calls):
                command = call.args[0]
                self.assertIn('tools/train.py' if index % 2 == 0 else 'tools/test.py', command)
                self.assertIn('randomness.seed=123', command)
                self.assertEqual('--no-pretrained' in command, index % 2 == 0)
                self.assertEqual(call.kwargs['cwd'], runner.ROOT)
                self.assertTrue(call.kwargs['check'])
                self.assertEqual(call.kwargs['env']['PYTHONPATH'].split(os.pathsep)[0], str(runner.ROOT))
                if index % 2:
                    self.assertTrue(command[3].endswith('best_binary_Foreground_IoU_iter_1000.pth'))
                    self.assertEqual(command[-4:-2], ['--split', 'test'])

    def test_checkpoint_selection_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(FileNotFoundError):
                runner.best_checkpoint(output, 'wwtp', False)
            (output / 'iter_20000.pth').touch()
            older = output / 'best_binary_Foreground_IoU_iter_1000.pth'
            newer = output / 'best_binary_Foreground_IoU_iter_2000.pth'
            older.touch()
            newer.touch()
            os.utime(older, ns=(1, 1))
            os.utime(newer, ns=(2, 2))
            self.assertEqual(runner.best_checkpoint(output, 'wwtp', False), newer)
            public = output / 'best_mIoU_iter_1000.pth'
            public.touch()
            self.assertEqual(runner.best_checkpoint(output, 'loveda', False), public)

    def test_preflight_all_directories_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'wwtp/hdnet'
            output.mkdir(parents=True)
            (output / 'last_checkpoint').touch()
            with patch.object(runner.subprocess, 'run') as execute:
                with self.assertRaises(FileExistsError):
                    runner.main(['train', '--models', 'hrnet', 'hdnet', '--work-root', directory])
                execute.assert_not_called()
            calls = self.run_main(['train', '--models', 'hdnet', '--work-root', directory,
                                   '--resume', '--gpus', '2']).call_args_list
            self.assertIn('--resume', calls[0].args[0])
            self.assertIn('--nproc-per-node=2', calls[0].args[0])

    def test_training_failure_does_not_evaluate_or_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'train')) as execute:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                    runner.main(['train-test', '--models', 'hrnet', 'hdnet', '--work-root', directory])
                self.assertEqual(execute.call_count, 1)

    def test_success_without_best_checkpoint_does_not_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner.subprocess, 'run') as execute:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileNotFoundError):
                    runner.main(['train-test', '--models', 'hrnet', '--work-root', directory])
                self.assertEqual(execute.call_count, 1)

    def test_public_validation_and_unsupported_suites(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'loveda/hrnet'
            output.mkdir(parents=True)
            (output / 'best_mIoU_iter_1000.pth').touch()
            command = self.run_main(['test', '--dataset', 'loveda', '--models', 'hrnet',
                                     '--work-root', directory]).call_args.args[0]
            self.assertEqual(command[-2:], ['--split', 'val'])
        for args in (['train', '--dataset', 'uavid', '--suite', 'v6-ablations'],
                     ['train', '--dataset', 'loveda', '--suite', 'all'],
                     ['test', '--dataset', 'uavid', '--split', 'test'],
                     ['train', '--suite', 'v6', '--models', 'hrnet'],
                     ['train', '--gpus', '0']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(args)

    def test_geometry_missing_complete_and_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pseudo = root / 'custom_geometry'
            for split in ('train', 'val', 'test'):
                (root / 'images' / split).mkdir(parents=True)
                (root / 'images' / split / 'sample.png').touch()
                (pseudo / split).mkdir(parents=True)
            jobs = {(str(root), str(pseudo), 'images'): {'train', 'val', 'test'}}
            with patch.object(runner, 'geometry_jobs', return_value=jobs), patch.object(
                    runner.subprocess, 'run') as execute, contextlib.redirect_stdout(io.StringIO()):
                runner.prepare_geometry([], 'wwtp', ['train', 'val', 'test'], [], False)
                self.assertEqual(execute.call_count, 1)
                command = execute.call_args.args[0]
                self.assertIn(str(pseudo), command)
                self.assertIn('tools/data/generate_pseudo_geometry.py', command)
                for split in ('train', 'val', 'test'):
                    (pseudo / split / 'sample.npz').touch()
                execute.reset_mock()
                runner.prepare_geometry([], 'wwtp', ['train', 'val', 'test'], [], False)
                execute.assert_not_called()
                (root / 'images/test/sample.png').unlink()
                with self.assertRaises(FileNotFoundError):
                    runner.prepare_geometry([], 'wwtp', ['train', 'val', 'test'], [], False)

    def test_geometry_reads_effective_pipelines_and_deduplicates(self):
        class OverridesAction(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                setattr(namespace, self.dest, dict(value.split('=', 1) for value in values))

        cfg = {
            f'{split}_dataloader': types.SimpleNamespace(dataset=types.SimpleNamespace(
                data_root='/actual/data', data_prefix=types.SimpleNamespace(img_path=f'images/{split}'),
                pipeline=[dict(type='LoadPseudoGeometry', pseudo_root='/actual/cache')]))
            for split in ('train', 'val', 'test')
        }
        merge = Mock()
        class EffectiveConfig(dict):
            merge_from_dict = merge
        config_module = types.ModuleType('mmengine.config')
        config_module.Config = types.SimpleNamespace(fromfile=Mock(return_value=EffectiveConfig(cfg)))
        config_module.DictAction = OverridesAction
        with patch.dict(sys.modules, {'mmengine': types.ModuleType('mmengine'), 'mmengine.config': config_module}):
            jobs = runner.geometry_jobs(['full.py', 'ablation.py'], 'wwtp', ['train', 'val', 'test'],
                                        ['randomness.seed=123'], False)
        self.assertEqual(jobs, {('/actual/data', '/actual/cache', 'images'): {'train', 'val', 'test'}})
        self.assertEqual(config_module.Config.fromfile.call_count, 2)
        merge.assert_called_with({'randomness.seed': '123'})

    def test_geometry_preparation_once_before_v6_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            def train(command, **kwargs):
                if 'tools/train.py' in command:
                    output = Path(command[command.index('--work-dir') + 1])
                    output.mkdir(parents=True)
                    (output / 'best_binary_Foreground_IoU_iter_1000.pth').touch()
            with patch.object(runner, 'prepare_geometry') as prepare:
                calls = self.run_main(['train-test', '--suite', 'v6-ablations',
                                       '--work-root', directory], train).call_args_list
            self.assertEqual(len(calls), 10)
            prepare.assert_called_once()
            self.assertEqual(len(prepare.call_args.args[0]), 5)
            self.assertEqual(prepare.call_args.args[2], ['train', 'val', 'test'])

    def test_dry_geometry_honors_environment_and_splits(self):
        with patch.dict(os.environ, {'WWTP_DATA_ROOT': '/custom/data', 'WWTP_PSEUDO_ROOT': '/custom/cache'}):
            self.assertEqual(runner.geometry_jobs([], 'wwtp', ['train', 'val'], [], True),
                             {('/custom/data', '/custom/cache', 'images'): {'train', 'val'}})
        with patch.dict(os.environ, {'UAVID_DATA_ROOT': '/public/data', 'PUBLIC_PSEUDO_ROOT': '/public/cache'}):
            self.assertEqual(runner.geometry_jobs([], 'uavid', ['train', 'val'], [], True),
                             {('/public/data', '/public/cache', 'img_dir'): {'train', 'val'}})

    def test_gate_v7_superiority_aborts_ablations_when_inferior(self):
        with tempfile.TemporaryDirectory() as directory:
            work_root = Path(directory)
            v7_eval = work_root / 'wwtp' / 'rpgv_v7' / 'eval'
            v7_eval.mkdir(parents=True)
            (v7_eval / 'metrics.json').write_text('{"binary/Foreground_IoU": 75.0}')

            baseline_eval = work_root / 'wwtp' / 'hrnet' / 'eval'
            baseline_eval.mkdir(parents=True)
            (baseline_eval / 'metrics.json').write_text('{"binary/Foreground_IoU": 76.5}')

            with contextlib.redirect_stdout(io.StringIO()) as stdout, patch.object(
                    runner.subprocess, 'run') as execute:
                runner.main(['train-test', '--suite', 'v7-ablations', '--work-root', directory,
                             '--no-auto-prerequisites'])
            self.assertEqual(execute.call_count, 0)
            self.assertIn('[GATE] RPGV v7 test metric (75.0000) is NOT higher than best baseline hrnet (76.5000)',
                          stdout.getvalue())

    def test_gate_v7_superiority_proceeds_when_superior(self):
        with tempfile.TemporaryDirectory() as directory:
            work_root = Path(directory)
            v7_eval = work_root / 'wwtp' / 'rpgv_v7' / 'eval'
            v7_eval.mkdir(parents=True)
            (v7_eval / 'metrics.json').write_text('{"binary/Foreground_IoU": 79.0}')

            baseline_eval = work_root / 'wwtp' / 'hrnet' / 'eval'
            baseline_eval.mkdir(parents=True)
            (baseline_eval / 'metrics.json').write_text('{"binary/Foreground_IoU": 76.5}')

            def train(command, **kwargs):
                if 'tools/train.py' in command:
                    output = Path(command[command.index('--work-dir') + 1])
                    output.mkdir(parents=True)
                    (output / 'best_binary_Foreground_IoU_iter_1000.pth').touch()

            with patch.object(runner, 'prepare_geometry'):
                calls = self.run_main(['train-test', '--suite', 'v7-ablations',
                                       '--work-root', directory, '--no-auto-prerequisites'], train).call_args_list
            self.assertEqual(len(calls), 12)

    def test_v7_ablations_auto_runs_v8_and_fresh_v7_prerequisites(self):
        with tempfile.TemporaryDirectory() as directory:
            work_root = Path(directory)
            baseline_eval = work_root / 'wwtp' / 'hrnet' / 'eval'
            baseline_eval.mkdir(parents=True)
            (baseline_eval / 'metrics.json').write_text('{"binary/Foreground_IoU": 75.0}')

            def fake_run(command, **kwargs):
                cmd_str = ' '.join(str(c) for c in command)
                if '--models rpgv_v8' in cmd_str:
                    v8_dir = work_root / 'wwtp' / 'rpgv_v8' / 'eval'
                    v8_dir.mkdir(parents=True, exist_ok=True)
                    (v8_dir / 'metrics.json').write_text('{"binary/Foreground_IoU": 82.0}')
                elif '--suite v7' in cmd_str and '--overwrite' in cmd_str:
                    v7_dir = work_root / 'wwtp' / 'rpgv_v7' / 'eval'
                    v7_dir.mkdir(parents=True, exist_ok=True)
                    (v7_dir / 'metrics.json').write_text('{"binary/Foreground_IoU": 81.0}')
                elif 'tools/train.py' in cmd_str:
                    out = Path(command[command.index('--work-dir') + 1])
                    out.mkdir(parents=True, exist_ok=True)
                    (out / 'best_binary_Foreground_IoU_iter_1000.pth').touch()

            with patch.object(runner, 'prepare_geometry'):
                calls = self.run_main(['train-test', '--suite', 'v7-ablations',
                                       '--work-root', directory], fake_run).call_args_list
            # Pre-requisite v8 + v7 retrain (2 top calls) + 12 ablation calls (6 models x 2)
            self.assertEqual(len(calls), 14)
            self.assertTrue((work_root / 'wwtp' / 'rpgv_v7' / '.retrained_30k_done').is_file())
            self.assertTrue((work_root / 'wwtp' / 'rpgv_v8' / 'eval' / 'metrics.json').is_file())

    def test_deepglobe_nine_models_dependency_free_dry_run(self):
        models = runner.MAINSTREAM_MODELS + ('rpgv_v6', 'rpgv_v7')
        for name in models:
            self.assertTrue(runner.experiment_config('deepglobe', name).is_file())
        result = subprocess.run([
            sys.executable, '-S', str(runner.ROOT / 'scripts/run_experiments.py'),
            'train-test', '--dataset', 'deepglobe', '--models', *models, '--dry-run'],
            capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.count('tools/train.py'), 9)
        self.assertEqual(result.stdout.count('tools/test.py'), 9)
        self.assertEqual(result.stdout.count('tools/data/generate_pseudo_geometry.py'), 1)
        self.assertEqual(result.stdout.count('--split test'), 9)
        self.assertIn('best_roadIoU_iter_SELECTED.pth', result.stdout)
        self.assertIn('--splits test train val', result.stdout)
        with self.assertRaises(ValueError):
            runner.experiment_config('deepglobe', 'rpgv_v7_fixed_grid')
        self.assertEqual(runner.extract_primary_metric({'roadIoU': 63.5, 'mIoU': 81.0}, 'deepglobe'), 63.5)
        self.assertIsNone(runner.extract_primary_metric({'mIoU': 81.0}, 'deepglobe'))

    def test_deepglobe_validation_selected_local_test(self):
        for split in (None, 'val'):
            with tempfile.TemporaryDirectory() as directory:
                def train(command, **kwargs):
                    if 'tools/train.py' in command:
                        output = Path(command[command.index('--work-dir') + 1])
                        output.mkdir(parents=True)
                        (output / 'best_roadIoU_iter_1000.pth').touch()
                        (output / 'iter_30000.pth').touch()
                args = ['train-test', '--dataset', 'deepglobe', '--models', 'hrnet',
                        '--work-root', directory, '--cfg-options', 'randomness.seed=123']
                if split:
                    args += ['--split', split]
                calls = self.run_main(args, train).call_args_list
                self.assertEqual(len(calls), 2)
                self.assertIn('best_roadIoU_iter_1000.pth', calls[1].args[0][3])
                self.assertEqual(calls[1].args[0][-4:-2], ['--split', split or 'test'])
                for call in calls:
                    self.assertIn('randomness.seed=123', call.args[0])

    def test_deepglobe_missing_best_and_failed_train_never_test(self):
        for error in (None, subprocess.CalledProcessError(1, 'train')):
            with tempfile.TemporaryDirectory() as directory:
                def train(command, **kwargs):
                    if error:
                        raise error
                    output = Path(command[command.index('--work-dir') + 1])
                    output.mkdir(parents=True)
                    (output / 'iter_30000.pth').touch()
                with contextlib.redirect_stdout(io.StringIO()), patch.object(
                        runner.subprocess, 'run', side_effect=train) as execute:
                    with self.assertRaises(subprocess.CalledProcessError if error else FileNotFoundError):
                        runner.main(['train-test', '--dataset', 'deepglobe', '--models', 'hrnet',
                                     '--work-root', directory])
                    self.assertEqual(execute.call_count, 1)

    def test_deepglobe_rpgv_geometry_splits(self):
        for mode, expected in (('train', ['train', 'val']), ('train-test', ['train', 'val', 'test'])):
            with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'prepare_geometry') as geometry:
                def train(command, **kwargs):
                    if 'tools/train.py' in command:
                        output = Path(command[command.index('--work-dir') + 1])
                        output.mkdir(parents=True)
                        (output / 'best_roadIoU_iter_1000.pth').touch()
                calls = self.run_main([mode, '--dataset', 'deepglobe', '--models', 'rpgv_v6', 'rpgv_v7',
                                       '--work-root', directory], train).call_args_list
                geometry.assert_called_once()
                self.assertEqual(len(geometry.call_args.args[0]), 2)
                self.assertEqual(geometry.call_args.args[2], expected)
                self.assertEqual(len(calls), 2 if mode == 'train' else 4)
        with patch.dict(os.environ, {'DEEPGLOBE_DATA_ROOT': '/roads', 'PUBLIC_PSEUDO_ROOT': '/geometry'}):
            self.assertEqual(runner.geometry_jobs([], 'deepglobe', ['test'], [], True),
                             {('/roads', '/geometry', 'img_dir'): {'test'}})

    def test_overwrite_clears_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'wwtp' / 'hrnet'
            output.mkdir(parents=True)
            dummy_file = output / 'dummy.txt'
            dummy_file.write_text('old')

            with self.assertRaises(FileExistsError):
                self.run_main(['train', '--models', 'hrnet', '--work-root', directory])

            def mock_train(command, **kwargs):
                out = Path(command[command.index('--work-dir') + 1])
                out.mkdir(parents=True, exist_ok=True)
                (out / 'new.txt').write_text('new')

            self.run_main(['train', '--models', 'hrnet', '--work-root', directory, '--overwrite'], mock_train)
            self.assertFalse(dummy_file.exists())
            self.assertTrue((output / 'new.txt').exists())


if __name__ == '__main__':
    unittest.main()
