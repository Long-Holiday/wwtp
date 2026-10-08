"""Configuration, label semantics, and official split leakage regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from mmengine.config import Config
from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.utils import register_all_modules

import wwtpseg  # noqa: F401
from tools.data.prepare_public import OFFICIAL_COUNTS, UAVID_COLORS, inventory, prepare, uavid_label_ids, _save_tiles
from tools.evaluation.predict_public import submission_mask
from wwtpseg.datasets.public_protocol import OFFICIAL_COUNTS as RUNTIME_COUNTS, validate_prepared
from wwtpseg.experiments import DATASETS as DATASET_NAMES, MAINSTREAM_MODELS, ROOT, experiment_config


class ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)

    def test_train_test_roster_and_budget(self):
        self.assertEqual(len(MAINSTREAM_MODELS), 7)
        self.assertEqual(RUNTIME_COUNTS, OFFICIAL_COUNTS)
        for dataset in DATASET_NAMES:
            for name in MAINSTREAM_MODELS:
                with self.subTest(dataset=dataset, model=name):
                    cfg = Config.fromfile(experiment_config(dataset, name))
                    self.assertEqual(cfg.model.data_preprocessor.size, (1024, 1024))
                    self.assertEqual(cfg.train_dataloader.batch_size * cfg.optim_wrapper.accumulative_counts, 8)
                    self.assertEqual(cfg.train_cfg.max_iters // cfg.optim_wrapper.accumulative_counts, 5000)
                    self.assertFalse(cfg.enable_early_stopping)
                    self.assertEqual(cfg.param_scheduler[-1].end, cfg.train_cfg.max_iters)
                    steps = {s.type for s in cfg.train_dataloader.dataset.pipeline}
                    self.assertFalse(steps & {'Resize', 'RandomResize', 'RandomCrop', 'RandomForegroundCrop'})
                    self.assertIn('RequireImageShape', steps)
                    if dataset == 'wwtp':
                        self.assertEqual(Path(cfg.data_root).name, 'wwtp_semantic_dataset_1m')
                        self.assertEqual(cfg.model.test_cfg.mode, 'whole')
                        boundary = next(x for x in cfg.test_evaluator if x.type == 'BinaryBoundaryMetric')
                        self.assertEqual((boundary.boundary_tolerance, boundary.pixel_size_m), (1.5, 1.0))
                    else:
                        self.assertEqual(cfg.test_dataloader, cfg.val_dataloader)
                        self.assertEqual(cfg.default_hooks.checkpoint.save_best, 'mIoU')
                        self.assertNotIn('LoadAnnotations', [s.type for s in cfg.predict_pipeline])
                        classes = 8 if dataset == 'uavid' else 7
                        if name == 'hdnet':
                            self.assertEqual(cfg.model.num_classes, classes)
                        else:
                            self.assertEqual(cfg.model.decode_head.num_classes, classes)
                        if 'auxiliary_head' in cfg.model:
                            self.assertEqual(cfg.model.auxiliary_head.num_classes, classes)
                        if name == 'mask2former':
                            self.assertEqual(len(cfg.model.decode_head.loss_cls.class_weight), classes + 1)

    def test_every_retained_config_resolves(self):
        for directory in ('wwtp', 'uavid', 'loveda', 'rpgv'):
            self.assertTrue((ROOT / 'configs' / directory).is_dir())
            for path in (ROOT / 'configs' / directory).rglob('*.py'):
                with self.subTest(config=str(path)):
                    cfg = Config.fromfile(path)
                    self.assertEqual(cfg.model.data_preprocessor.size, (1024, 1024))
                    self.assertNotIn('Potsdam', cfg.train_dataloader.dataset.type)

    def test_binary_and_multiclass_pipeline_label_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in DATASET_NAMES:
                cfg = Config.fromfile(experiment_config(dataset, 'segformer'))
                data = copy.deepcopy(cfg.train_dataloader.dataset)
                data.data_root = str(root / dataset)
                for prefix in data.data_prefix.values():
                    (Path(data.data_root) / prefix).mkdir(parents=True)
                Image.new('RGB', (1024, 1024), (32, 64, 96)).save(Path(data.data_root) / data.data_prefix.img_path / 'sample.png')
                labels = np.tile(np.arange(1024, dtype=np.uint16) % (2 if dataset == 'wwtp' else 8), (1024, 1)).astype(np.uint8)
                Image.fromarray(labels).save(Path(data.data_root) / data.data_prefix.seg_map_path / 'sample.png')
                sample = DATASETS.build(data)[0]
                observed = set(sample['data_samples'].gt_sem_seg.data.unique().tolist())
                self.assertEqual(observed, {0, 1} if dataset == 'wwtp' else set(range(8)) if dataset == 'uavid' else set(range(7)) | {255})
                self.assertEqual(tuple(sample['inputs'].shape), (3, 1024, 1024))
        reject = TRANSFORMS.build(dict(type='RequireImageShape', size=(1024, 1024)))
        with self.assertRaises(ValueError):
            reject(dict(img=np.zeros((2048, 2048, 3)), img_path='old.png'))

    def test_uavid_color_encoding_unknown_colors_and_border_tiles(self):
        colors = np.asarray([list(UAVID_COLORS) + [(255, 255, 255)]], dtype=np.uint8)
        labels = np.asarray(uavid_label_ids(Image.fromarray(colors)))
        np.testing.assert_array_equal(labels[0], list(range(8)) + [255])
        with self.assertRaises(ValueError):
            uavid_label_ids(Image.new('RGB', (1, 1), (1, 2, 3)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'img_dir/train').mkdir(parents=True)
            (root / 'ann_dir/train').mkdir(parents=True)
            count = _save_tiles(Image.new('RGB', (1025, 1025), 'white'), Image.new('L', (1025, 1025), 3), root, 'train', 'seq1_000000')
            self.assertEqual(count, 4)
            corner = np.asarray(Image.open(root / 'ann_dir/train/seq1_000000_x1024_y1024.png'))
            self.assertEqual(corner[0, 0], 3)
            self.assertTrue(np.all(corner[1:] == 255))

    def test_private_test_submission_encoding(self):
        ids = np.arange(7, dtype=np.uint8)[None]
        path, mask = submission_mask('LoveDADataset', 'Urban_1234.png', ids)
        self.assertEqual(path, Path('1234.png'))
        np.testing.assert_array_equal(np.asarray(mask), ids)
        ids = np.arange(8, dtype=np.uint8)[None]
        path, mask = submission_mask('UAVidDataset', 'seq22_000100.png', ids)
        self.assertEqual(path, Path('seq22/Labels/000100.png'))
        np.testing.assert_array_equal(np.asarray(mask)[0], UAVID_COLORS)

    def test_exact_loveda_counts_and_untouched_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offset = 0
            for split, count in OFFICIAL_COUNTS['loveda'].items():
                for domain in ('Urban', 'Rural'):
                    (root / split.capitalize() / domain / 'images_png').mkdir(parents=True)
                    if split != 'test':
                        (root / split.capitalize() / domain / 'masks_png').mkdir()
                for index in range(count):
                    domain = 'Urban' if index % 2 else 'Rural'
                    base = root / split.capitalize() / domain
                    name = f'{offset + index}.png'
                    (base / 'images_png' / name).touch()
                    if split != 'test':
                        (base / 'masks_png' / name).touch()
                offset += count
            data = inventory('loveda', root)
            self.assertEqual({k: len(v) for k, v in data.items()}, OFFICIAL_COUNTS['loveda'])
            data['val'][0].unlink()
            with self.assertRaisesRegex(ValueError, '1669'):
                prepare('loveda', root, root / 'prepared')
            self.assertFalse((root / 'prepared.building').exists())

    def test_sequence_split_counts_and_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            next_seq = 1
            for split, count in OFFICIAL_COUNTS['uavid'].items():
                for _ in range(count // 10):
                    group = root / f'uavid_{split}' / f'seq{next_seq}'
                    (group / 'Images').mkdir(parents=True)
                    if split != 'test':
                        (group / 'Labels').mkdir()
                    for frame in range(10):
                        (group / 'Images' / f'{frame:06d}.png').touch()
                        if split != 'test':
                            (group / 'Labels' / f'{frame:06d}.png').touch()
                    next_seq += 1
            data = inventory('uavid', root)
            self.assertEqual({k: len(v) for k, v in data.items()}, OFFICIAL_COUNTS['uavid'])
            prepared = root / 'prepared'
            manifest = dict(dataset='uavid', patch_size=1024,
                            official_source_counts=OFFICIAL_COUNTS['uavid'], counts={},
                            source_files={}, sequence_splits={})
            for split, paths in data.items():
                (prepared / 'img_dir' / split).mkdir(parents=True)
                if split != 'test':
                    (prepared / 'ann_dir' / split).mkdir(parents=True)
                for path in paths:
                    name = f'{path.parent.parent.name}_{path.stem}'
                    if split == 'train':
                        name += '_x0000_y0000'
                    (prepared / 'img_dir' / split / f'{name}.png').touch()
                    if split != 'test':
                        (prepared / 'ann_dir' / split / f'{name}.png').touch()
                manifest['counts'][split] = dict(source_images=len(paths), prepared_images=len(paths))
                manifest['source_files'][split] = [str(p.relative_to(root)) for p in paths]
                manifest['sequence_splits'][split] = sorted({p.parent.parent.name for p in paths})
            (prepared / 'manifest.json').write_text(json.dumps(manifest))
            validate_prepared('uavid', prepared)
            original = next((prepared / 'img_dir/test').glob('*.png'))
            original.rename(original.with_name('seq36_999999.png'))
            with self.assertRaisesRegex(ValueError, 'frame identities'):
                validate_prepared('uavid', prepared)
            first_val_seq = data['val'][0].parent.parent
            first_val_seq.rename(first_val_seq.with_name('seq1'))
            with self.assertRaisesRegex(ValueError, 'sequence leakage'):
                inventory('uavid', root)


if __name__ == '__main__':
    unittest.main()
