"""TensorBoard service started alongside a training run.

MMEngine's ``TensorboardVisBackend`` already writes event files for every run;
this module only brings up the browser-facing service so training exposes its
curves without a separate command. The service is a detached process: it keeps
serving after a run finishes, and it never affects training if it cannot start.
"""
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from mmengine.logging import print_log

DEFAULT_PORT = 6006
DEFAULT_HOST = '0.0.0.0'


@dataclass(frozen=True)
class TensorBoardService:
    """Handle of a TensorBoard service that was started by this process."""

    url: str
    logdir: str
    log_path: Path
    process: subprocess.Popen


def is_main_process() -> bool:
    """Whether the caller is rank 0 of a distributed launch."""
    return os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')) == '0'


def default_logdir(work_dir) -> str:
    """Directory TensorBoard should watch for a run stored in ``work_dir``.

    Runs are created as ``<root>/<dataset>/<model>/<timestamp>``, so watching
    the top-level component of the work dir keeps every experiment started from
    the same working directory on one page. A work dir outside the working
    directory is watched on its own.
    """
    path = Path(work_dir).resolve()
    cwd = Path.cwd().resolve()
    try:
        relative = path.relative_to(cwd)
    except ValueError:
        return str(path)
    return str(cwd / relative.parts[0]) if len(relative.parts) > 1 else str(path)


def _port_is_open(host: str, port: int) -> bool:
    probe = socket.socket()
    probe.settimeout(0.5)
    try:
        return probe.connect_ex((host, port)) == 0
    finally:
        probe.close()


def start_tensorboard_service(work_dir, port: int = DEFAULT_PORT, logdir=None,
                              host: str = DEFAULT_HOST, startup_timeout: float = 15.0):
    """Start a detached TensorBoard serving ``work_dir`` and return its handle.

    Returns ``None`` instead of raising when the port already hosts a service,
    when TensorBoard is not installed, or when it fails to come up; the caller
    keeps training either way.
    """
    if not is_main_process():
        return None
    logdir = str(Path(logdir).resolve()) if logdir else default_logdir(work_dir)
    probe_host = '127.0.0.1' if host in ('0.0.0.0', '::') else host
    if _port_is_open(probe_host, port):
        print_log(
            f'TensorBoard port {port} is already serving, not starting a second one; event files '
            f'keep going to {work_dir}. Use --tensorboard-port for a separate service.',
            logger='current')
        return None
    log_path = Path(work_dir) / f'tensorboard-{port}.log'
    command = [sys.executable, '-m', 'tensorboard.main', '--logdir', logdir,
               '--host', host, '--port', str(port), '--reload_interval', '5']
    try:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        Path(logdir).mkdir(parents=True, exist_ok=True)
        with open(log_path, 'wb') as log_file:
            process = subprocess.Popen(
                command, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as error:
        print_log(f'Could not launch TensorBoard ({error}); training continues, '
                  f'event files keep going to {work_dir}.', logger='current')
        return None
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline and process.poll() is None:
        if _port_is_open(probe_host, port):
            url = f'http://localhost:{port}/'
            print_log(f'TensorBoard: {url} (logdir: {logdir})', logger='current')
            return TensorBoardService(url=url, logdir=logdir, log_path=log_path, process=process)
        time.sleep(0.25)
    print_log(f'TensorBoard did not come up on port {port}{_failure_reason(process, log_path)}; '
              f'training continues and event files keep going to {work_dir}. If this environment '
              f'lacks TensorBoard, install it as in requirements/runtime.txt.', logger='current')
    return None


def _failure_reason(process: subprocess.Popen, log_path: Path) -> str:
    """Last line the failed TensorBoard subprocess wrote, for the warning."""
    if process.poll() is None:
        return ' within the startup timeout'
    try:
        lines = [line.strip() for line in log_path.read_text(errors='replace').splitlines()
                 if line.strip()]
    except OSError:
        return ''
    return f': {lines[-1]}' if lines else ''
