#!/usr/bin/env python3
"""Finish the eight WWTP baselines to their configured 20k-iteration budget.

Run inside the project container after other GPU training has finished:
    python scripts/complete_mainstream_baselines.py --dry-run
    python scripts/complete_mainstream_baselines.py

The script resumes the latest iteration checkpoint, including optimizer and
scheduler state. When an extended run has not started, it resumes the original
3k checkpoint with the extended scheduler instead. It never uses test results
to decide whether training is complete or which weights to select.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT / "work_dirs"
TARGET_ITERS = 20_000
INITIAL_ITERS = 3_000
MODELS = (
    "deeplabv3plus",
    "hrnet",
    "mask2former",
    "rs_mamba",
    "segformer",
    "segnext",
    "unet",
    "unetformer",
)
ITER_CHECKPOINT = re.compile(r"iter_(\d+)\.pth\Z")
BEST_CHECKPOINT = re.compile(r"best_binary_Foreground_IoU_iter_(\d+)\.pth\Z")


def iteration(path: Path) -> int | None:
    match = ITER_CHECKPOINT.fullmatch(path.name)
    return int(match.group(1)) if match else None


def complete_checkpoint(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def latest_iteration_checkpoint(directory: Path) -> Path | None:
    """Prefer MMEngine's completed-checkpoint pointer over a stray partial file."""
    pointer = directory / "last_checkpoint"
    if pointer.is_file():
        pointed = directory / Path(pointer.read_text().strip()).name
        if iteration(pointed) is not None and complete_checkpoint(pointed):
            return pointed
    candidates = [
        path for path in directory.glob("iter_*.pth")
        if iteration(path) is not None and complete_checkpoint(path)
    ]
    return max(candidates, key=lambda path: iteration(path) or -1) if candidates else None


def configured_target(config: Path) -> int:
    text = config.read_text(encoding="utf-8")
    match = re.search(r"train_cfg\s*=\s*dict\([^\n]*max_iters\s*=\s*(\d+)", text)
    if not match:
        raise RuntimeError(f"Cannot verify max_iters in {config}")
    return int(match.group(1))


def plan_model(model: str, python_bin: str) -> tuple[str, Path | None, list[str] | None]:
    config = ROOT / "configs" / "experiments_extended" / f"{model}.py"
    if configured_target(config) != TARGET_ITERS:
        raise RuntimeError(f"{config} does not target {TARGET_ITERS} iterations")

    extended = WORK_ROOT / f"{model}_extended"
    final = extended / f"iter_{TARGET_ITERS}.pth"
    if complete_checkpoint(final):
        return "complete", final, None

    source = latest_iteration_checkpoint(extended)
    fresh_extension = source is None
    if fresh_extension:
        # An existing extended run without a resumable iteration checkpoint
        # needs inspection; silently restarting it could overwrite its best.
        if extended.is_dir() and any(extended.iterdir()):
            raise RuntimeError(f"{extended} has results but no resumable iter_*.pth")
        source = WORK_ROOT / model / f"iter_{INITIAL_ITERS}.pth"
        if not complete_checkpoint(source):
            raise FileNotFoundError(f"Missing original checkpoint: {source}")

    step = iteration(source)
    if step is None or step >= TARGET_ITERS:
        raise RuntimeError(f"Unexpected source checkpoint: {source}")

    command = [
        python_bin, "tools/train.py", str(config.relative_to(ROOT)),
        "--work-dir", str(extended.relative_to(ROOT)),
        "--resume", str(source.relative_to(ROOT)),
        "--disable-early-stopping",
    ]
    if fresh_extension:
        command.append("--no-resume-scheduler")
    return f"resume {step}/{TARGET_ITERS}", source, command


def validation_score(directory: Path, step: int) -> float | None:
    scores = []
    for path in directory.glob("*/vis_data/scalars.json"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (record.get("step") == step
                    and "binary/Foreground_IoU" in record):
                scores.append(float(record["binary/Foreground_IoU"]))
    return max(scores) if scores else None


def best_validation_checkpoint(model: str) -> tuple[Path, float] | None:
    candidates = []
    for directory in (WORK_ROOT / model, WORK_ROOT / f"{model}_extended"):
        for path in directory.glob("best_binary_Foreground_IoU_iter_*.pth"):
            match = BEST_CHECKPOINT.fullmatch(path.name)
            if not match or not complete_checkpoint(path):
                continue
            score = validation_score(directory, int(match.group(1)))
            if score is not None:
                candidates.append((path, score))
    return max(candidates, key=lambda item: item[1]) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show resume plans without training")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS,
                        help="train only selected models; default: all eight")
    args = parser.parse_args()

    for model in args.models:
        status, source, command = plan_model(model, sys.executable)
        print(f"[{model}] {status}; checkpoint: {source}", flush=True)
        if command is None:
            continue
        print(f"  {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=ROOT, check=True)
        final = WORK_ROOT / f"{model}_extended" / f"iter_{TARGET_ITERS}.pth"
        if not complete_checkpoint(final):
            raise RuntimeError(f"[{model}] Training exited without {final}")
        print(f"[{model}] completed {TARGET_ITERS} iterations", flush=True)

    if not args.dry_run:
        print("\nValidation-selected checkpoints (original and extended runs):")
        for model in args.models:
            best = best_validation_checkpoint(model)
            if best is None:
                print(f"  {model}: no readable validation-selected checkpoint")
            else:
                path, score = best
                print(f"  {model}: {score:.4f}%  {path.relative_to(ROOT)}")
        print("Existing test results and inference gallery remain tied to their old checkpoints.")


if __name__ == "__main__":
    main()
