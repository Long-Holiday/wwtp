"""Unit tests for SegNeXt-MSCAN-S 1024 / 1m training protocol."""

import os
import unittest
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.structures import PixelData
from mmseg.registry import DATASETS, MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401


class SegNeXt1mTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        torch.set_num_threads(2)

    def test_config_values(self):
        configs = [
            'configs/experiments/segnext.py',
            'configs/v5/controls/segnext_1m.py',
        ]
        for path in configs:
            with self.subTest(config=path):
                cfg = Config.fromfile(path)
                self.assertIn('1m', cfg.data_root)
                self.assertEqual(cfg.data_root, 'wwtp_semantic_dataset_1m')
                self.assertEqual(cfg.model.data_preprocessor.size, (1024, 1024))
                self.assertEqual(cfg.model.test_cfg.mode, 'whole')
                self.assertEqual(cfg.train_cfg.max_iters, 20000)
                self.assertEqual(cfg.train_cfg.val_interval, 1000)
                self.assertEqual(cfg.optim_wrapper.accumulative_counts, 4)
                self.assertEqual(cfg.optim_wrapper.type, 'OptimWrapper')
                self.assertEqual(cfg.train_dataloader.batch_size, 2)
                self.assertFalse(cfg.enable_early_stopping)
                self.assertEqual(cfg.work_dir, 'work_dirs/segnext_1m')

    def test_pipeline_and_evaluator_spec(self):
        cfg = Config.fromfile('configs/experiments/segnext.py')
        for split in ('train', 'val', 'test'):
            dataset_cfg = cfg[f'{split}_dataloader'].dataset
            self.assertEqual(dataset_cfg.data_root, 'wwtp_semantic_dataset_1m')
            steps = [step['type'] for step in dataset_cfg.pipeline]
            self.assertIn('PackSegInputs', steps)
            self.assertFalse({'RandomCrop', 'Resize', 'RandomResize', 'RandomForegroundCrop'} & set(steps))

        evaluator_types = [m['type'] for m in cfg.val_evaluator]
        self.assertEqual(evaluator_types, ['IoUMetric', 'BinaryBoundaryMetric', 'BinaryTopologyMetric'])
        boundary_metric = next(m for m in cfg.val_evaluator if m['type'] == 'BinaryBoundaryMetric')
        self.assertEqual(boundary_metric['boundary_tolerance'], 1.5)
        self.assertEqual(boundary_metric['pixel_size_m'], 1.0)
        topology_metric = next(m for m in cfg.val_evaluator if m['type'] == 'BinaryTopologyMetric')
        self.assertEqual(topology_metric['small_area'], 64)

    def test_data_isolation_and_historical_integrity(self):
        cfg = Config.fromfile('configs/experiments/segnext.py')
        # Ensure segnext does not write to original baseline directories
        self.assertNotEqual(cfg.work_dir, 'work_dirs/segnext')
        self.assertNotEqual(cfg.work_dir, 'work_dirs/segnext_extended')

        # Ensure experiments_extended retains its original 512 dataset configuration
        ext_cfg = Config.fromfile('configs/experiments_extended/segnext.py')
        self.assertIn('512', ext_cfg.train_dataloader.dataset.pipeline[2].crop_size
                      if hasattr(ext_cfg.train_dataloader.dataset.pipeline[2], 'crop_size')
                      else '(512, 512)')
        self.assertEqual(ext_cfg.model.data_preprocessor.size, (512, 512))

    def test_model_build_forward_loss_cpu(self):
        cfg = Config.fromfile('configs/experiments/segnext.py')
        cfg.model.backbone.init_cfg = None

        model = MODELS.build(cfg.model)
        model.train()

        img = torch.randn(1, 3, 128, 128)
        gt_sem_seg = torch.randint(0, 2, (1, 128, 128)).long()

        sample = SegDataSample()
        sample.set_metainfo(dict(
            img_shape=(128, 128),
            ori_shape=(128, 128),
            pad_shape=(128, 128),
            scale_factor=(1.0, 1.0)
        ))
        sample.gt_sem_seg = PixelData(data=gt_sem_seg)

        losses = model.loss(img, [sample])
        self.assertIn('decode.loss_ce', losses)
        self.assertIn('decode.loss_dice', losses)
        total, _ = model.parse_losses(losses)
        self.assertTrue(torch.isfinite(total))
        total.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f'Parameter {name} has no gradient')
                self.assertTrue(torch.isfinite(param.grad).all(), f'Gradient for {name} is not finite')

    def test_real_sample_loading(self):
        cfg = Config.fromfile('configs/experiments/segnext.py')
        dataset = DATASETS.build(cfg.train_dataloader.dataset)
        self.assertEqual(len(dataset), 2435)

        sample = dataset[0]
        self.assertEqual(sample['inputs'].shape, (3, 1024, 1024))
        self.assertEqual(sample['data_samples'].gt_sem_seg.data.shape, (1, 1024, 1024))


if __name__ == '__main__':
    unittest.main()
