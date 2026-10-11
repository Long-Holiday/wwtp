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
                         {'baselines': 7, 'v6': 1, 'v6-ablations': 5, 'all': 13})
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


if __name__ == '__main__':
    unittest.main()
