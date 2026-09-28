"""Checkpoint selection for fixed-budget WWTP baseline completion."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import complete_mainstream_baselines as completion


class CompletionPlanTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = self.root / "configs/experiments_extended/segformer.py"
        config.parent.mkdir(parents=True)
        config.write_text(
            "train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)\n"
        )
        self.work_root = self.root / "work_dirs"
        self.work_root.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(completion, "ROOT", self.root).start()
        patch.object(completion, "WORK_ROOT", self.work_root).start()

    def test_resumes_extended_checkpoint_with_scheduler(self):
        work_dir = self.work_root / "segformer_extended"
        work_dir.mkdir()
        (work_dir / "iter_7000.pth").write_bytes(b"complete checkpoint")
        (work_dir / "iter_8000.pth").write_bytes(b"partial checkpoint")
        (work_dir / "last_checkpoint").write_text(
            "/workspace/work_dirs/segformer_extended/iter_7000.pth")

        status, source, command = completion.plan_model("segformer", "python")

        self.assertEqual(status, "resume 7000/20000")
        self.assertEqual(source.name, "iter_7000.pth")
        self.assertIn("--disable-early-stopping", command)
        self.assertNotIn("--no-resume-scheduler", command)

    def test_starts_extension_from_original_with_new_scheduler(self):
        original = self.work_root / "segformer"
        original.mkdir()
        (original / "iter_3000.pth").write_bytes(b"complete checkpoint")

        status, source, command = completion.plan_model("segformer", "python")

        self.assertEqual(status, "resume 3000/20000")
        self.assertEqual(source.name, "iter_3000.pth")
        self.assertIn("--no-resume-scheduler", command)

    def test_test_results_do_not_count_as_finished_training(self):
        work_dir = self.work_root / "segformer_extended"
        (work_dir / "test_results").mkdir(parents=True)
        (work_dir / "test_results/old.json").write_text("{}")
        (work_dir / "iter_10000.pth").write_bytes(b"complete checkpoint")

        status, _, command = completion.plan_model("segformer", "python")

        self.assertEqual(status, "resume 10000/20000")
        self.assertIsNotNone(command)

    def test_skips_only_when_final_iteration_checkpoint_exists(self):
        work_dir = self.work_root / "segformer_extended"
        work_dir.mkdir()
        final = work_dir / "iter_20000.pth"
        final.write_bytes(b"complete checkpoint")

        status, source, command = completion.plan_model("segformer", "python")

        self.assertEqual((status, source, command), ("complete", final, None))


if __name__ == "__main__":
    unittest.main()
