"""DeepGlobe preparation/runtime regression tests without the training stack."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.data import prepare_public as preparation
from tools.data.deepglobe_protocol import local_splits, source_id
from tools.evaluation.predict_public import submission_mask

ROOT = Path(__file__).resolve().parents[1]
runtime = runpy.run_path(str(ROOT / 'wwtpseg/datasets/public_protocol.py'))
validate_prepared = runtime['validate_prepared']


class DeepGlobePreparationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'raw'
        self.output = self.root / 'prepared'
        counts = {'train': 20, 'valid': 2, 'test': 3}
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(preparation.OFFICIAL_COUNTS, deepglobe=counts))
        stack.enter_context(patch.dict(runtime['OFFICIAL_COUNTS'], deepglobe=counts))
        self.make_source('train', 20)

    def make_source(self, split, count):
        directory = self.source / split
        directory.mkdir(parents=True)
        for index in range(count):
            identifier = f'{split}_{index:04d}'
            Image.new('RGB', (1024, 1024), (40, 80, 120)).save(directory / f'{identifier}_sat.jpg')
            if split == 'train':
                values = np.zeros((1024, 1024), dtype=np.uint8)
                values[:, 500:520] = 255
                image = Image.fromarray(values)
                if index % 2:
                    image = image.convert('RGB')
                image.save(directory / f'{identifier}_mask.png')

    def prepare(self, **kwargs):
        return preparation.prepare('deepglobe', self.source, self.output, **kwargs)

    def test_labeled_train_only_and_white_is_road(self):
        manifest = self.prepare()
        self.assertEqual(manifest, validate_prepared('deepglobe', self.output))
        self.assertEqual(manifest['prediction_source_files'], {})
        self.assertEqual({s: c['prepared_images'] for s, c in manifest['counts'].items()},
                         {'train': 16, 'val': 2, 'test': 2})
        self.assertEqual(manifest['checkpoint_selection_split'], 'val')
        self.assertEqual(manifest['local_evaluation_split'], 'test')
        for split in ('train', 'val', 'test'):
            images = sorted((self.output / 'img_dir' / split).glob('*.png'))
            for path in images:
                with Image.open(self.output / 'ann_dir' / split / path.name) as image:
                    self.assertEqual(set(np.unique(np.asarray(image))), {0, 1})
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_repeatable_custom_splits(self):
        first = self.prepare()
        second = preparation.prepare('deepglobe', self.source, self.root / 'second')
        self.assertEqual(first['source_files'], second['source_files'])
        custom = preparation.prepare('deepglobe', self.source, self.root / 'custom',
                                     split_seed=7, split_ratios=(0.6, 0.2, 0.2))
        self.assertNotEqual(first['source_files'], custom['source_files'])
        self.assertEqual(custom['counts']['val']['source_images'], 4)
        validate_prepared('deepglobe', self.root / 'custom')
        sets = [set(files) for files in custom['source_files'].values()]
        self.assertEqual(len(set.union(*sets)), 20)
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_optional_official_predictions_are_separate(self):
        self.make_source('valid', 2)
        self.make_source('test', 3)
        manifest = self.prepare()
        validate_prepared('deepglobe', self.output)
        self.assertEqual(set(manifest['prediction_source_files']), {'official_val', 'official_test'})
        self.assertEqual(manifest['counts']['official_test']['prepared_images'], 3)
        self.assertEqual(manifest['counts']['test']['prepared_images'], 2)
        self.assertFalse((self.output / 'ann_dir' / 'official_test').exists())

    def test_bad_image_mask_inventory_before_writing(self):
        mask = self.source / 'train/train_0000_mask.png'
        mask.unlink()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name('prepared.building').exists())
        Image.new('L', (1024, 1024)).save(mask)
        Image.new('L', (1024, 1024)).save(mask.with_name('orphan_mask.png'))
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.prepare()

    def test_bad_optional_inventory_and_cross_source_duplicate(self):
        self.make_source('valid', 1)
        with self.assertRaisesRegex(ValueError, 'expected 2'):
            preparation.inventory('deepglobe', self.source)
        Image.new('RGB', (1024, 1024)).save(self.source / 'valid/train_0000_sat.jpg')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            preparation.inventory('deepglobe', self.source)

    def test_invalid_pixels_and_shapes(self):
        for values in (np.full((4, 4), 1, dtype=np.uint8),
                       np.full((4, 4, 3), [0, 255, 0], dtype=np.uint8),
                       np.zeros((4, 4, 4), dtype=np.uint8)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                preparation.deepglobe_label_ids(Image.fromarray(values))
        Image.new('L', (16, 16), 255).save(self.source / 'train/train_0000_mask.png')
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertTrue(self.output.with_name('prepared.building').is_dir())
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_wrong_image_size(self):
        Image.new('RGB', (32, 32)).save(self.source / 'train/train_0000_sat.jpg')
        with self.assertRaisesRegex(ValueError, '1024x1024'):
            self.prepare()

    def test_manifest_tampering_rejected(self):
        original = self.prepare()
        cases = []
        seed = copy.deepcopy(original)
        seed['split_seed'] = 100
        cases.append(seed)
        membership = copy.deepcopy(original)
        membership['source_files']['train'][0], membership['source_files']['val'][0] = (
            membership['source_files']['val'][0], membership['source_files']['train'][0])
        cases.append(membership)
        duplicate = copy.deepcopy(original)
        duplicate['source_files']['test'][0] = duplicate['source_files']['train'][0]
        cases.append(duplicate)
        official = copy.deepcopy(original)
        official['source_files']['val'][0] = 'valid/fake_sat.jpg'
        cases.append(official)
        count = copy.deepcopy(original)
        count['counts']['test']['source_images'] += 1
        cases.append(count)
        for manifest in cases:
            with self.subTest(manifest=manifest):
                (self.output / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    validate_prepared('deepglobe', self.output)
        (self.output / 'manifest.json').write_text(json.dumps(original))
        next((self.output / 'ann_dir/test').glob('*.png')).unlink()
        with self.assertRaisesRegex(ValueError, 'image/mask'):
            validate_prepared('deepglobe', self.output)

    def test_missing_manifest_and_wrong_prepared_identity(self):
        with self.assertRaises(FileNotFoundError):
            validate_prepared('deepglobe', self.output)
        self.prepare()
        path = next((self.output / 'img_dir/train').glob('*.png'))
        path.rename(path.with_name('wrong.png'))
        with self.assertRaisesRegex(ValueError, 'identities'):
            validate_prepared('deepglobe', self.output)

    def test_split_inputs_and_standard_rounding(self):
        files = [f'train/{i}_sat.jpg' for i in range(6226)]
        self.assertEqual({s: len(f) for s, f in local_splits(files).items()},
                         {'train': 4982, 'val': 622, 'test': 622})
        for ratios in ((0.8, 0.2, 0), (0.8, 0.1, 0.2), (float('nan'), 0.1, 0.1)):
            with self.assertRaises(ValueError):
                local_splits(files, ratios=ratios)
        for filename in ('../12_sat.jpg', '/train/12_sat.jpg', 'train/../12_sat.jpg', 'valid/12_sat.jpg'):
            with self.assertRaises(ValueError):
                source_id(filename)
        with self.assertRaises(ValueError):
            local_splits(['train/1_sat.jpg'] * 20)

    def test_submission_encoding_and_id(self):
        ids = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        path, image = submission_mask('DeepGlobeRoadDataset', '123.png', ids)
        self.assertEqual(path, Path('123_mask.png'))
        np.testing.assert_array_equal(np.asarray(image), ids * 255)
        path, _ = submission_mask('DeepGlobeRoadDataset', 'id_with_underscores.png', ids)
        self.assertEqual(path.name, 'id_with_underscores_mask.png')
        with self.assertRaises(ValueError):
            submission_mask('DeepGlobeRoadDataset', '123.png', np.array([[255]]))


if __name__ == '__main__':
    unittest.main()
