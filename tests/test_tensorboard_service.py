"""The TensorBoard service training starts must come up, reuse ports and never block training."""
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

from wwtpseg.engine.tensorboard_service import (
    TensorBoardService, default_logdir, is_main_process, start_tensorboard_service)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class TensorBoardServiceTest(unittest.TestCase):
    def test_default_logdir_watches_the_experiments_root(self):
        root = Path.cwd().resolve()
        self.assertEqual(default_logdir('work_dirs/wwtp/hrnet'), str(root / 'work_dirs'))
        self.assertEqual(default_logdir('work_dirs/rpgv_v6_siglip_1m'), str(root / 'work_dirs'))
        outside = Path(tempfile.mkdtemp()) / 'run'
        self.assertEqual(default_logdir(outside), str(outside))

    def test_only_rank_zero_starts_the_service(self):
        self.assertTrue(is_main_process())
        with patch.dict(os.environ, {'RANK': '1'}):
            self.assertFalse(is_main_process())
            self.assertIsNone(start_tensorboard_service('work_dirs/wwtp/hrnet', port=free_port()))

    def test_service_serves_the_run_and_leaves_an_occupied_port_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            port = free_port()
            service = start_tensorboard_service(
                Path(directory) / 'run', port=port, logdir=directory, host='127.0.0.1')
            self.assertIsNotNone(service, 'TensorBoard did not start; is it installed in this image?')
            self.addCleanup(service.process.wait)
            self.addCleanup(service.process.terminate)
            self.assertIsInstance(service, TensorBoardService)
            self.assertEqual(service.logdir, str(Path(directory).resolve()))
            with urlopen(service.url, timeout=15) as page:
                self.assertIn(b'TensorBoard', page.read())
            second = start_tensorboard_service(
                Path(directory) / 'second-run', port=port, logdir=directory, host='127.0.0.1')
            self.assertIsNone(second, 'a run must not start a second service on an occupied port')


if __name__ == '__main__':
    unittest.main()
