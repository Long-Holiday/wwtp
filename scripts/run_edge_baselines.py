#!/usr/bin/env python3
"""Automated runner for CBR-Net and HD-Net edge baseline experiments.

Executes training in Docker, evaluates best checkpoints on the test set,
and writes comprehensive benchmark results to work_dirs/edge_baselines/summary.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_ROOT = REPO_ROOT / "work_dirs/edge_baselines"
MODELS = ["cbr_net", "hd_net"]


def run_cmd(cmd: list[str] | str, check: bool = True) -> int:
    display_cmd = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"\n[EXEC] {display_cmd}\n")
    sys.stdout.flush()
    res = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=str(REPO_ROOT))
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {display_cmd}")
    return res.returncode


def get_best_ckpt(model_dir: str | Path) -> str | None:
    ckpts = sorted(glob.glob(os.path.join(str(model_dir), "best_binary_Foreground_IoU_iter_*.pth")))
    if ckpts:
        return ckpts[-1]
    ckpts = sorted(glob.glob(os.path.join(str(model_dir), "iter_*.pth")))
    if ckpts:
        return ckpts[-1]
    return None


def get_best_val(model_dir: str | Path) -> dict | None:
    scalars_files = sorted(glob.glob(os.path.join(str(model_dir), "**", "vis_data", "scalars.json"), recursive=True))
    best_record = None
    best_iou = -1.0
    for sf in scalars_files:
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    if "binary/Foreground_IoU" in d and "step" in d:
                        iou = float(d["binary/Foreground_IoU"])
                        if iou > best_iou:
                            best_iou = iou
                            best_record = d
                except Exception:
                    pass
    return best_record


def get_test_eval(model_dir: str | Path) -> dict | None:
    eval_files = sorted(glob.glob(os.path.join(str(model_dir), "test_eval", "**", "*.json"), recursive=True))
    for ef in reversed(eval_files):
        try:
            with open(ef, "r", encoding="utf-8") as f:
                d = json.load(f)
                if "binary/Foreground_IoU" in d:
                    return d
        except Exception:
            pass
    return None


def update_summary(work_root: Path) -> None:
    if not work_root.is_dir():
        return

    rows = []
    for model_name in MODELS:
        model_dir = work_root / model_name
        if not model_dir.is_dir():
            continue
        val_rec = get_best_val(model_dir)
        test_rec = get_test_eval(model_dir)

        row = {
            "Model": model_name,
            "Val_Iter": str(val_rec.get("step", "-")) if val_rec else "-",
            "Val_IoU": f"{val_rec['binary/Foreground_IoU']:.2f}%" if val_rec else "-",
            "Val_Dice": f"{val_rec['binary/Dice']:.2f}%" if val_rec else "-",
            "Val_HD95_m": f"{val_rec.get('binary/HD95_m', val_rec.get('binary/HD95_px', 0.0) * 0.5):.1f}m" if val_rec else "-",
            "Test_IoU": f"{test_rec['binary/Foreground_IoU']:.2f}%" if test_rec else "-",
            "Test_Dice": f"{test_rec['binary/Dice']:.2f}%" if test_rec else "-",
            "Test_HD95_m": f"{test_rec.get('binary/HD95_m', test_rec.get('binary/HD95_px', 0.0) * 0.5):.1f}m" if test_rec else "-",
            "Test_Boundary_F1": f"{test_rec.get('binary/Boundary_F1', test_rec.get('binary/BFScore', 0.0)):.2f}%" if test_rec else "-",
        }
        rows.append(row)

    if not rows:
        return

    headers = [
        "Model", "Val_Iter", "Val_IoU", "Val_Dice", "Val_HD95_m",
        "Test_IoU", "Test_Dice", "Test_HD95_m", "Test_Boundary_F1"
    ]
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    header_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"

    lines = [
        "# CBR-Net 与 HD-Net 边界对比模型评测汇总表\n",
        f"*更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
        header_line,
        sep_line,
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[h]).ljust(col_widths[h]) for h in headers) + " |")

    summary_text = "\n".join(lines) + "\n"
    summary_path = work_root / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("\n" + summary_text)


def run_baseline(
    model_name: str,
    work_root: Path = DEFAULT_WORK_ROOT,
    dry_run: bool = False,
) -> None:
    config = f"configs/edge_baselines/{model_name}.py"
    config_path = REPO_ROOT / config
    if not config_path.is_file():
        raise FileNotFoundError(f"Config {config} does not exist!")

    model_dir = work_root / model_name
    test_eval_dir = model_dir / "test_eval"
    training_done = model_dir / ".training_completed"

    print(f"\n{'='*70}\n[START] Edge Baseline: {model_name}\nConfig: {config}\nWorkDir: {model_dir}\n{'='*70}")

    best_ckpt = get_best_ckpt(model_dir)
    test_res = get_test_eval(model_dir)

    if best_ckpt and test_res:
        print(f"[SKIP] {model_name} already finished training & evaluation.")
        print(f"       Best Checkpoint: {best_ckpt}")
        print(f"       Test IoU: {test_res.get('binary/Foreground_IoU', 0.0):.2f}%")
        update_summary(work_root)
        return

    if dry_run:
        print(f"[DRY-RUN] Verified config {config} exists. Target work dir: {model_dir}")
        return

    model_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Training
    if not training_done.is_file():
        print(f"\n>>> [Stage 1/2] Launching Training for {model_name}...")
        rel_model_dir = model_dir.relative_to(REPO_ROOT)
        train_cmd = [
            "docker", "compose", "run", "--rm",
            "-e", "PYTHONUNBUFFERED=1",
            "wwtp", "python", "tools/train.py", config,
            "--work-dir", str(rel_model_dir)
        ]
        run_cmd(train_cmd)
        best_ckpt = get_best_ckpt(model_dir)
        if not best_ckpt:
            raise RuntimeError(f"No checkpoint produced in {model_dir} after training!")
        training_done.write_text(f"config={config}\ncompleted_at={time.ctime()}\n", encoding="utf-8")

    if not best_ckpt:
        best_ckpt = get_best_ckpt(model_dir)
    if not best_ckpt:
        raise RuntimeError(f"No checkpoint produced in {model_dir} after training!")

    print(f"\n[OK] Training completed. Best checkpoint: {best_ckpt}")

    # Step 2: Test set evaluation
    if not test_res:
        print(f"\n>>> [Stage 2/2] Launching Test Set Evaluation for {model_name}...")
        test_eval_dir.mkdir(parents=True, exist_ok=True)
        rel_test_eval_dir = test_eval_dir.relative_to(REPO_ROOT)
        rel_best_ckpt = Path(best_ckpt).relative_to(REPO_ROOT)
        test_cmd = [
            "docker", "compose", "run", "--rm",
            "-e", "PYTHONUNBUFFERED=1",
            "wwtp", "python", "tools/test.py", config, str(rel_best_ckpt),
            "--work-dir", str(rel_test_eval_dir)
        ]
        run_cmd(test_cmd)

    update_summary(work_root)
    print(f"\n[DONE] Baseline {model_name} completed successfully!\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS + ["all"],
        default=["all"],
        help="Edge baseline models to train and evaluate (default: all)",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=DEFAULT_WORK_ROOT,
        help="Directory to save baseline logs and checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configurations without running training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    selected_models = MODELS if "all" in args.models else args.models
    print(f"Edge Baseline Suite: {len(selected_models)} models scheduled: {selected_models}")

    for model_name in selected_models:
        run_baseline(model_name, work_root=work_root, dry_run=args.dry_run)

    update_summary(work_root)
    print("\n[ALL COMPLETE] All scheduled edge baseline models finished.")


if __name__ == "__main__":
    main()
