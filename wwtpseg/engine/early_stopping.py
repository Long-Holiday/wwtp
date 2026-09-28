"""Resume-aware early stopping using the validation history in the run directory."""
import json
import math
from pathlib import Path

from mmengine.hooks import EarlyStoppingHook
from mmengine.registry import HOOKS


def validation_history(work_dir, monitor, through_iter=None):
    # Resumes may repeat a validation iteration; use its most recent record.
    rows = {}
    for path in sorted(Path(work_dir).glob('*/vis_data/scalars.json')):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            step = record.get('iter', record.get('step'))
            if monitor not in record or not isinstance(step, (int, float)):
                continue
            if through_iter is not None and step > through_iter:
                continue
            score = record[monitor]
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise FloatingPointError(f'{path}: non-finite validation metric {monitor}')
            rows[int(step)] = dict(record, selected_iter=int(step))
    return [rows[step] for step in sorted(rows)]


def plateau_reached(rows, options):
    hook = EarlyStoppingHook(**{k: v for k, v in options.items() if k != 'type'})
    stopped = False
    for row in rows:
        stopped, _ = hook._check_stop_condition(row[hook.monitor])
    return stopped


@HOOKS.register_module()
class RPGVEarlyStoppingHook(EarlyStoppingHook):
    """Keep patience across resumes and fail, rather than succeed, on NaN/Inf.

    Iteration checkpoints are written before validation. Replaying logged
    validations through the restored iteration also recovers that iteration's
    patience update, which would otherwise be absent from the checkpoint.
    """

    def before_train(self, runner):
        if runner.iter == 0:
            return
        rows = validation_history(runner.work_dir, self.monitor, runner.iter)
        if not rows:
            return
        self.wait_count = 0
        self.best_score = -math.inf if self.rule == 'greater' else math.inf
        stopped = False
        for row in rows:
            stopped, _ = self._check_stop_condition(row[self.monitor])
        runner.train_loop.stop_training = stopped

    def after_val_epoch(self, runner, metrics):
        if self.monitor in metrics and not math.isfinite(metrics[self.monitor]):
            raise FloatingPointError(f'Non-finite validation metric: {self.monitor}')
        super().after_val_epoch(runner, metrics)
