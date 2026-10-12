"""Training-stack checks for DeepGlobe configuration, metrics and RPGV smoke paths."""
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.structures import PixelData
from mmseg.registry import METRICS
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401
from tests.smoke_test import smoke_model
from tools.data import prepare_public as preparation
from tools.evaluation import predict_public
from wwtpseg.datasets import public_protocol
from wwtpseg.engine.runtime import prepare_runtime
from wwtpseg.experiments import MAINSTREAM_MODELS, experiment_config


class DeepGlobeTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        register_all_modules(init_default_scope=True)

    def test_nine_configs_and_shared_strategy(self):
        reference = Config.fromfile(experiment_config('loveda', 'hrnet'))
        for name in MAINSTREAM_MODELS + ('rpgv_v6', 'rpgv_v7'):
            cfg = Config.fromfile(experiment_config('deepglobe', name))
            with self.subTest(model=name):
                self.assertEqual(cfg.benchmark_dataset, 'deepglobe')
                self.assertEqual(cfg.default_hooks.checkpoint.save_best, 'roadIoU')
                self.assertEqual(cfg.model.test_cfg.mode, 'whole')
                classes = cfg.model.get('num_classes', cfg.model.get('decode_head', {}).get('num_classes'))
                self.assertEqual(classes, 2)
                for field in ('optim_wrapper', 'param_scheduler', 'train_cfg', 'randomness', 'env_cfg'):
                    self.assertEqual(cfg[field], reference[field])
                rgb_steps = [s for s in cfg.train_dataloader.dataset.pipeline if s['type'] != 'LoadPseudoGeometry']
                self.assertEqual(rgb_steps, reference.train_pipeline)
                for split in ('train', 'val', 'test'):
                    data = cfg[f'{split}_dataloader'].dataset
                    self.assertEqual(data.type, 'DeepGlobeRoadDataset')
                    self.assertEqual(data.data_prefix.img_path, f'img_dir/{split}')
                    self.assertEqual(data.data_prefix.seg_map_path, f'ann_dir/{split}')
                self.assertNotIn('LoadAnnotations', [s.type for s in cfg.predict_pipeline])
                if name.startswith('rpgv'):
                    for split in ('train', 'val', 'test'):
                        pipeline = cfg[f'{split}_dataloader'].dataset.pipeline
                        geometry = next(s for s in pipeline if s.type == 'LoadPseudoGeometry')
                        self.assertTrue(geometry.required)

    def test_environment_roots(self):
        with patch.dict(os.environ, {'DEEPGLOBE_DATA_ROOT': '/roads', 'PUBLIC_PSEUDO_ROOT': '/geometry'}):
            cfg = Config.fromfile(experiment_config('deepglobe', 'rpgv_v7'))
        self.assertEqual(cfg.data_root, '/roads')
        for split in ('train', 'val', 'test'):
            data = cfg[f'{split}_dataloader'].dataset
            self.assertEqual(data.data_root, '/roads')
            self.assertEqual(next(s.pseudo_root for s in data.pipeline if s.type == 'LoadPseudoGeometry'), '/geometry')

    def evaluate_masks(self, targets, predictions):
        metric = METRICS.build(dict(type='RoadIoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']))
        metric.dataset_meta = dict(classes=('background', 'road'))
        for target, prediction in zip(targets, predictions):
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=torch.tensor(target, dtype=torch.long).unsqueeze(0))
            sample.pred_sem_seg = PixelData(data=torch.tensor(prediction, dtype=torch.long).unsqueeze(0))
            metric.process({}, [sample.to_dict()])
        return metric.evaluate(len(targets))

    def test_road_iou_f1_ignore_and_empty(self):
        perfect = self.evaluate_masks([[[0, 1], [255, 1]]], [[[0, 1], [1, 1]]])
        self.assertEqual(perfect['roadIoU'], 100.0)
        self.assertEqual(perfect['roadF1'], 100.0)
        self.assertEqual(perfect['mIoU'], 100.0)
        self.assertNotIn('iou/roadIoU', perfect)
        missed = self.evaluate_masks([[[0, 1], [0, 1]]], [[[0, 0], [0, 0]]])
        self.assertEqual(missed['roadIoU'], 0.0)
        self.assertEqual(missed['roadF1'], 0.0)
        empty = self.evaluate_masks([[[0, 0]]], [[[0, 0]]])
        self.assertEqual(empty['roadIoU'], 0.0)
        self.assertEqual(empty['roadF1'], 0.0)
        all_ignore = self.evaluate_masks([[[255, 255]]], [[[0, 1]]])
        self.assertEqual(all_ignore['roadIoU'], 0.0)

    def test_road_metric_aggregates_pixels_not_image_ious(self):
        metrics = self.evaluate_masks([[[1, 0, 0, 0]], [[1, 1, 1, 1]]],
                                      [[[1, 0, 0, 0]], [[0, 0, 0, 0]]])
        self.assertEqual(metrics['roadIoU'], 20.0)
        self.assertEqual(metrics['roadF1'], 33.33)
        self.assertIn('mDice', metrics)
        self.assertIn('mFscore', metrics)

    def test_prepared_data_runner_train_validation_checkpoint_and_test(self):
        counts = {'train': 10, 'valid': 2, 'test': 3}
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                preparation.OFFICIAL_COUNTS, deepglobe=counts), patch.dict(
                public_protocol.OFFICIAL_COUNTS, deepglobe=counts):
            root = Path(directory)
            source = root / 'raw/train'
            source.mkdir(parents=True)
            mask = np.zeros((1024, 1024), dtype=np.uint8)
            mask[:, 400:600] = 255
            for index in range(10):
                Image.new('RGB', (1024, 1024), (40, 80, 120)).save(source / f'{index}_sat.jpg')
                Image.fromarray(mask).save(source / f'{index}_mask.png')
            official = source.parent / 'test'
            official.mkdir()
            for index in range(3):
                Image.new('RGB', (1024, 1024), (40, 80, 120)).save(official / f'official{index}_sat.jpg')
            prepared = root / 'prepared'
            preparation.prepare('deepglobe', source.parent, prepared)
            cfg = Config.fromfile(experiment_config('deepglobe', 'segformer'))
            cfg.data_root = str(prepared)
            cfg.predict_dataset.data_root = str(prepared)
            cfg.work_dir = str(root / 'run')
            cfg.model.data_preprocessor.size = (64, 64)
            cfg.optim_wrapper.accumulative_counts = 1
            cfg.train_cfg.max_iters = 2
            cfg.train_cfg.val_interval = 2
            cfg.param_scheduler = [dict(type='PolyLR', begin=0, end=2, by_epoch=False, power=0.9)]
            cfg.default_hooks.checkpoint.interval = 2
            cfg.visualizer.name = 'deepglobe-runner-smoke'
            for split in ('train', 'val', 'test'):
                loader = cfg[f'{split}_dataloader']
                loader.num_workers = 0
                loader.persistent_workers = False
                loader.dataset.data_root = str(prepared)
                # Only the smoke run resizes to 64; production configs stay at 1024.
                resize = dict(type='Resize', scale=(64, 64), keep_ratio=False)
                middle = [dict(type='LoadAnnotations'), resize] if split == 'train' else [resize, dict(type='LoadAnnotations')]
                loader.dataset.pipeline = [dict(type='LoadImageFromFile'), *middle, dict(type='PackSegInputs')]
            prepare_runtime(cfg, no_pretrained=True)
            runner = Runner.from_cfg(cfg)
            runner.train()
            best = sorted((root / 'run').glob('best_roadIoU_iter_*.pth'))
            self.assertEqual(len(best), 1)
            self.assertTrue((root / 'run/iter_2.pth').is_file())
            runner.load_checkpoint(str(best[0]))
            scores = runner.test()
            self.assertIn('roadIoU', scores)
            self.assertIn('roadF1', scores)
            self.assertEqual(Path(runner.test_dataloader.dataset.data_prefix['img_path']), prepared / 'img_dir/test')
            runner.visualizer.close()
            cfg.predict_pipeline = [dict(type='LoadImageFromFile'),
                                    dict(type='Resize', scale=(64, 64), keep_ratio=False),
                                    dict(type='PackSegInputs')]
            config_path = root / 'smoke_config.py'
            cfg.dump(config_path)
            output = root / 'official_predictions'
            with patch.object(sys, 'argv', ['predict_public', str(config_path), str(best[0]),
                                           '--device', 'cpu', '--output', str(output)]):
                predict_public.main()
            self.assertEqual({p.name for p in output.glob('*.png')},
                             {f'official{i}_mask.png' for i in range(3)})
            for path in output.glob('*.png'):
                with Image.open(path) as image:
                    self.assertEqual(image.size, (1024, 1024))
                    self.assertTrue(set(np.unique(np.asarray(image))).issubset({0, 255}))

    def test_rpgv_v6_v7_binary_forward_backward(self):
        for name in ('rpgv_v6', 'rpgv_v7'):
            with self.subTest(model=name):
                smoke_model(name, experiment_config('deepglobe', name), 64, backward=True, device='cpu')


if __name__ == '__main__':
    unittest.main()
