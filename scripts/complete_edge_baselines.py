#!/usr/bin/env python3
"""Resume and complete edge baselines (CBR-Net and HD-Net) to their 20k-iteration budget.

Usage inside docker container or on host with docker:
    python scripts/complete_edge_baselines.py --dry-run
    python scripts/complete_edge_baselines.py
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT / "work_dirs" / "edge_baselines"
TARGET_ITERS = 20_000
MODELS = ["cbr_net", "hd_net"]
ITER_CHECKPOINT = re.compile(r"iter_(\d+)\.pth\Z")


def complete_checkpoint(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def iteration(path: Path) -> int | None:
    match = ITER_CHECKPOINT.fullmatch(path.name)
    return int(match.group(1)) if match else None


def latest_iteration_checkpoint(directory: Path) -> Path | None:
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


def get_best_ckpt(model_dir: Path) -> Path | None:
    ckpts = sorted(model_dir.glob("best_binary_Foreground_IoU_iter_*.pth"))
    if ckpts and complete_checkpoint(ckpts[-1]):
        return ckpts[-1]
    ckpts = sorted(model_dir.glob("iter_*.pth"))
    if ckpts and complete_checkpoint(ckpts[-1]):
        return ckpts[-1]
    return None


def run_cmd(cmd: list[str] | str, check: bool = True) -> int:
    display_cmd = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"\n[EXEC] {display_cmd}\n", flush=True)
    res = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=str(REPO_ROOT))
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {display_cmd}")
    return res.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show execution plan without running")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS,
                        help="models to complete (default: cbr_net hd_net)")
    args = parser.parse_args()

    # Import update_summary from run_edge_baselines
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from run_edge_baselines import update_summary
    except ImportError:
        update_summary = None

    for model in args.models:
        model_dir = WORK_ROOT / model
        config = f"configs/edge_baselines/{model}.py"
        config_path = REPO_ROOT / config
        if not config_path.is_file():
            raise FileNotFoundError(f"Config {config} not found")

        final_ckpt = model_dir / f"iter_{TARGET_ITERS}.pth"
        if complete_checkpoint(final_ckpt):
            print(f"[{model}] already completed {TARGET_ITERS} iterations (found {final_ckpt.relative_to(REPO_ROOT)})")
            continue

        latest_ckpt = latest_iteration_checkpoint(model_dir)
        if latest_ckpt is None:
            raise RuntimeError(f"[{model}] No resumable checkpoint found in {model_dir}")

        step = iteration(latest_ckpt)
        print(f"[{model}] Current progress: {step}/{TARGET_ITERS} iterations. Checkpoint: {latest_ckpt.relative_to(REPO_ROOT)}")

        # Clean old .training_completed flag if present so it doesn't block evaluation
        training_done_flag = model_dir / ".training_completed"
        if training_done_flag.is_file() and not args.dry_run:
            training_done_flag.unlink()

        # Step 1: Training to TARGET_ITERS with disabled early stopping
        train_cmd = [
            "python", "tools/train.py", config,
            "--work-dir", str(model_dir.relative_to(REPO_ROOT)),
            "--resume", str(latest_ckpt.relative_to(REPO_ROOT)),
            "--disable-early-stopping",
        ]
        print(f"  Training command: {shlex.join(train_cmd)}")

        if not args.dry_run:
            run_cmd(train_cmd)
            if not complete_checkpoint(final_ckpt):
                raise RuntimeError(f"[{model}] Training exited without producing {final_ckpt}")
            training_done_flag.write_text(f"config={config}\ncompleted_at={time.ctime()}\n", encoding="utf-8")
            print(f"[{model}] Successfully completed {TARGET_ITERS} iterations!")

        # Step 2: Evaluation on test set with best checkpoint
        best_ckpt = get_best_ckpt(model_dir)
        print(f"[{model}] Best checkpoint for test evaluation: {best_ckpt.relative_to(REPO_ROOT) if best_ckpt else 'None'}")
        if best_ckpt:
            test_eval_dir = model_dir / "test_eval"
            test_cmd = [
                "python", "tools/test.py", config, str(best_ckpt.relative_to(REPO_ROOT)),
                "--work-dir", str(test_eval_dir.relative_to(REPO_ROOT)),
            ]
            print(f"  Test evaluation command: {shlex.join(test_cmd)}")
            if not args.dry_run:
                run_cmd(test_cmd)

        if not args.dry_run and update_summary is not None:
            update_summary(WORK_ROOT)

    print("\n[DONE] Edge baselines completion plan finished.")


if __name__ == "__main__":
    main()
