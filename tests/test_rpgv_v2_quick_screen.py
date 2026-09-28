#!/usr/bin/env python3
"""Unit tests for the deliberately biased RPGV-v2 screening funnel."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from mmengine.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from run_rpgv_v2_quick_screen import (  # noqa: E402
    FULL,
    QuickBinaryOverlapMetric,
    evaluation_config,
    expand_variants,
    fast_finetune_config,
    official_command,
    rank_records,
    select_from_summary,
    stratified_selection,
)


class QuickScreenTest(unittest.TestCase):
    def test_group_expansion_is_stable_and_deduplicated(self):
        names = expand_variants(['progressive', 'progressive_dfgv'])
        self.assertEqual(names.count('progressive_dfgv'), 1)
        self.assertEqual(names[0], 'progressive_base')
        self.assertEqual(names[-1], 'progressive_contour')

    def test_stratified_manifest_is_fixed_and_balanced(self):
        rows = [dict(index=i, image=f'/image/{i}.png', annotation=f'/mask/{i}.png',
                     foreground_fraction=i / 100) for i in range(100)]
        first = stratified_selection(rows, 10, seed=7)
        second = stratified_selection(rows, 10, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len({row['index'] for row in first}), 10)
        counts = [sum(row['coverage_quartile'] == q for row in first) for q in range(4)]
        self.assertEqual(counts, [3, 3, 2, 2])

    def test_overlap_metric_and_ranking(self):
        metric = QuickBinaryOverlapMetric()
        values = metric.compute_metrics([dict(tp=8, fp=2, fn=2, tn=88)])
        self.assertAlmostEqual(values['Foreground_IoU'], 100 * 8 / 12)
        self.assertEqual(values['Dice'], 80.0)

        def record(name, score):
            return dict(identity=dict(variant=name), metrics={
                'binary/Foreground_IoU': score,
                'binary/Dice': score,
                'binary/Precision': score,
                'binary/Recall': score,
            })
        ranked = rank_records([record(FULL, 80), record('no_rgr', 70),
                               record('no_contour', 75)])
        self.assertEqual([row['identity']['variant'] for row in ranked],
                         [FULL, 'no_contour', 'no_rgr'])
        summary = dict(rows=[dict(variant=FULL), dict(variant='no_contour'),
                             dict(variant='no_rgr')])
        self.assertEqual(select_from_summary(summary, 1), ['no_contour'])

    def test_fast_config_changes_only_screening_protocol(self):
        manifest = dict(size=4, samples=[dict(index=i) for i in range(4)])
        with tempfile.TemporaryDirectory() as temporary:
            cfg = fast_finetune_config(
                'progressive_dfgv', Path(temporary) / 'full.pth',
                Path(temporary) / 'train', manifest,
                seed=42, iterations=3000, crop_size=512,
                thumbnail_size=256, resize_base=1024,
                train_batch_size=2, eval_batch_size=1, accumulation=1,
                workers=0, learning_rate=1e-4, checkpoint_interval=1000)
            generated = Path(temporary) / 'generated.py'
            generated.write_text(cfg.pretty_text, encoding='utf-8')
            round_trip = Config.fromfile(generated)
            evaluation = evaluation_config(
                generated, Path(temporary) / 'tuned.pth', [1, 3],
                Path(temporary) / 'eval', batch_size=1, workers=0)
        self.assertEqual(tuple(cfg.model.data_preprocessor.size), (512, 512))
        self.assertEqual(cfg.model.global_thumbnail_size, 256)
        self.assertEqual(cfg.train_cfg.max_iters, 3000)
        self.assertEqual(cfg.train_cfg.val_interval, 3000)
        self.assertEqual(cfg.optim_wrapper.accumulative_counts, 1)
        self.assertEqual(cfg.optim_wrapper.optimizer.lr, 1e-4)
        self.assertFalse(cfg.randomness.deterministic)
        self.assertTrue(cfg.env_cfg.cudnn_benchmark)
        self.assertFalse(cfg.enable_early_stopping)
        self.assertEqual(list(cfg.val_dataloader.dataset.indices), [0, 1, 2, 3])
        transforms = {item.type: item for item in cfg.train_dataloader.dataset.pipeline}
        self.assertEqual(tuple(transforms['RandomForegroundCrop'].crop_size), (512, 512))
        self.assertEqual(tuple(transforms['GenerateGlobalThumbnail'].size), (256, 256))
        self.assertEqual(tuple(transforms['RandomResize'].scale), (1024, 1024))
        self.assertEqual(round_trip.train_cfg.max_iters, 3000)
        self.assertEqual(tuple(round_trip.model.data_preprocessor.size), (512, 512))
        self.assertEqual(tuple(evaluation.model.data_preprocessor.size), (512, 512))
        self.assertEqual(evaluation.model.global_thumbnail_size, 256)
        self.assertEqual(list(evaluation.test_dataloader.dataset.indices), [1, 3])

    def test_official_command_is_portable(self):
        args = type('Args', (), dict(
            official_work_root=None, seed=42, official_max_iters=100000))()
        command = official_command(
            ['progressive_dfgv'], args,
            Path(__file__).resolve().parents[1] / 'work_dirs' / 'quick')
        self.assertEqual(command[0], 'python')
        self.assertEqual(command[1], 'scripts/run_rpgv_v2_ablations.py')
        self.assertIn('work_dirs/quick/official_confirmation', command)


if __name__ == '__main__':
    unittest.main()
