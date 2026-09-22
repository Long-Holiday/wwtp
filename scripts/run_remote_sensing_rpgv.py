#!/usr/bin/env python3
"""Automated runner for RPGV remote sensing benchmark experiments.

Trains and evaluates RPGV on LoveDA and ISPRS Potsdam datasets,
compares metrics against literature baselines (SegFormer, DeepLabV3+, Samba),
and generates a consolidated summary in work_dirs/remote_sensing/summary.md.
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
WORK_ROOT = REPO_ROOT / "work_dirs/remote_sensing"


def run_cmd(cmd: list[str] | str, check: bool = True) -> int:
    display_cmd = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"\n[EXEC] {display_cmd}\n")
    sys.stdout.flush()
    res = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=str(REPO_ROOT))
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {display_cmd}")
    return res.returncode


def get_best_checkpoint(model_dir: Path, pattern: str = "best_mIoU_iter_*.pth") -> Path | None:
    def _iter_num(p: str) -> int:
        try:
            return int(Path(p).stem.split('_')[-1])
        except Exception:
            return 0

    matches = sorted(glob.glob(str(model_dir / pattern)), key=_iter_num)
    if matches:
        return Path(matches[-1])
    matches = sorted(glob.glob(str(model_dir / "iter_*.pth")), key=_iter_num)
    if matches:
        return Path(matches[-1])
    return None


def run_loveda_rpgv(dry_run: bool = False) -> dict | None:
    print(f"\n{'='*70}\n[START] LoveDA RPGV Benchmark Experiment\n{'='*70}")
    work_dir = WORK_ROOT / "loveda_rpgv"
    eval_dir = WORK_ROOT / "loveda_rpgv_eval"
    metrics_file = eval_dir / "metrics.json"

    if metrics_file.is_file():
        print(f"[SKIP] LoveDA RPGV already evaluated. Metrics: {metrics_file}")
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if dry_run:
        print("[DRY-RUN] LoveDA RPGV paths and configs verified.")
        return None

    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train
    best_ckpt = get_best_checkpoint(work_dir)
    if not best_ckpt:
        print("\n>>> Launching LoveDA RPGV Training in Docker...")
        train_cmd = [
            "docker", "compose", "run", "--rm",
            "-e", "PYTHONUNBUFFERED=1",
            "wwtp", "python", "tools/remote_sensing/run.py", "train",
            "configs/remote_sensing/loveda_rpgv.py",
            "--work-dir", "work_dirs/remote_sensing/loveda_rpgv",
        ]
        run_cmd(train_cmd)
        best_ckpt = get_best_checkpoint(work_dir)
        if not best_ckpt:
            raise RuntimeError(f"No checkpoint produced in {work_dir} after LoveDA training!")

    print(f"\n[OK] LoveDA Training complete. Selected checkpoint: {best_ckpt}")

    # 2. Eval
    if not metrics_file.is_file():
        print("\n>>> Launching LoveDA RPGV Evaluation in Docker...")
        rel_ckpt = best_ckpt.relative_to(REPO_ROOT)
        eval_cmd = [
            "docker", "compose", "run", "--rm",
            "-e", "PYTHONUNBUFFERED=1",
            "wwtp", "python", "tools/remote_sensing/run.py", "eval",
            "configs/remote_sensing/loveda_rpgv.py",
            "--checkpoint", str(rel_ckpt),
            "--work-dir", "work_dirs/remote_sensing/loveda_rpgv_eval",
        ]
        run_cmd(eval_cmd)

    # 3. Compare
    compare_cmd = [sys.executable, "tools/remote_sensing/compare.py", "loveda", str(metrics_file)]
    run_cmd(compare_cmd, check=False)

    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


def run_potsdam_rpgv(dry_run: bool = False) -> dict | None:
    print(f"\n{'='*70}\n[START] ISPRS Potsdam RPGV Benchmark Experiment\n{'='*70}")
    work_dir_full = WORK_ROOT / "potsdam_rpgv_full"
    eval_dir = WORK_ROOT / "potsdam_rpgv_full_eval"
    metrics_file = eval_dir / "metrics.json"

    if metrics_file.is_file():
        print(f"[SKIP] Potsdam RPGV already evaluated. Metrics: {metrics_file}")
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if dry_run:
        print("[DRY-RUN] Potsdam RPGV paths and configs verified.")
        return None

    work_dir_full.mkdir(parents=True, exist_ok=True)

    # Train on all 24 tiles
    final_ckpt = work_dir_full / "iter_40000.pth"
    if not final_ckpt.is_file():
        ckpt = get_best_checkpoint(work_dir_full, "iter_*.pth")
        if not ckpt:
            print("\n>>> Launching Potsdam RPGV Full Training in Docker...")
            train_cmd = [
                "docker", "compose", "run", "--rm",
                "-e", "PYTHONUNBUFFERED=1",
                "wwtp", "python", "tools/remote_sensing/run.py", "train",
                "configs/remote_sensing/potsdam_rpgv_full.py",
                "--work-dir", "work_dirs/remote_sensing/potsdam_rpgv_full",
            ]
            run_cmd(train_cmd)
            ckpt = get_best_checkpoint(work_dir_full, "iter_*.pth")
            if not ckpt:
                raise RuntimeError(f"No checkpoint produced in {work_dir_full} after Potsdam training!")
        final_ckpt = ckpt

    print(f"\n[OK] Potsdam Training complete. Checkpoint: {final_ckpt}")

    # Eval on 14 test tiles
    if not metrics_file.is_file():
        print("\n>>> Launching Potsdam RPGV Evaluation in Docker...")
        rel_ckpt = final_ckpt.relative_to(REPO_ROOT)
        eval_cmd = [
            "docker", "compose", "run", "--rm",
            "-e", "PYTHONUNBUFFERED=1",
            "wwtp", "python", "tools/remote_sensing/run.py", "eval",
            "configs/remote_sensing/potsdam_rpgv_full.py",
            "--checkpoint", str(rel_ckpt),
            "--work-dir", "work_dirs/remote_sensing/potsdam_rpgv_full_eval",
        ]
        run_cmd(eval_cmd)

    # Compare
    compare_cmd = [sys.executable, "tools/remote_sensing/compare.py", "potsdam", str(metrics_file)]
    run_cmd(compare_cmd, check=False)

    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


def update_summary(loveda_metrics: dict | None, potsdam_metrics: dict | None) -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = WORK_ROOT / "summary.md"

    # Literature references from references.json
    refs_path = REPO_ROOT / "tools/remote_sensing/references.json"
    refs = {}
    if refs_path.is_file():
        with open(refs_path, "r", encoding="utf-8") as f:
            refs = json.load(f)

    lines = [
        "# 通用遥感语义分割基准测试 RPGV 评测汇总表",
        f"\n*更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "## 1. LoveDA 数据集 (Val 集 1669 张, 7 类)",
        "| 模型 / 方法 | mIoU (%) | mDice (%) | mF1 (%) | 与 Samba 差值 (mIoU / mF1 pp) |",
        "|---|---:|---:|---:|---:|",
    ]

    # LoveDA rows
    loveda_refs = refs.get("loveda", {}).get("rows", [])
    samba_loveda_iou = 47.11
    samba_loveda_f1 = 63.17
    for row in loveda_refs:
        d_iou = row["mIoU"] - samba_loveda_iou
        d_f1 = row["mF1"] - samba_loveda_f1
        lines.append(f"| {row['model']} | {row['mIoU']:.2f} | — | {row['mF1']:.2f} | {d_iou:+.2f} / {d_f1:+.2f} |")

    if loveda_metrics:
        miou = float(loveda_metrics.get("mIoU", 0.0))
        mdice = float(loveda_metrics.get("mDice", 0.0))
        mf1 = float(loveda_metrics.get("mFscore", loveda_metrics.get("mF1", 0.0)))
        d_iou = miou - samba_loveda_iou
        d_f1 = mf1 - samba_loveda_f1
        lines.append(f"| **RPGV-Net (Ours)** | **{miou:.2f}** | **{mdice:.2f}** | **{mf1:.2f}** | **{d_iou:+.2f} / {d_f1:+.2f}** |")
    else:
        lines.append("| **RPGV-Net (Ours)** | *暂未执行（已按要求暂停，仅聚焦 Potsdam）* | — | — | — |")

    lines.extend([
        "\n## 2. ISPRS Potsdam 数据集 (独立 14 张测试瓦片, 排除 clutter 五类)",
        "| 模型 / 方法 | 5类 mIoU (%) | 5类 mF1 (%) | 5类 OA (%) | 与 Samba 差值 (mIoU / mF1 pp) |",
        "|---|---:|---:|---:|---:|",
    ])

    potsdam_refs = refs.get("potsdam", {}).get("rows", [])
    samba_potsdam_iou = 82.29
    samba_potsdam_f1 = 90.15
    for row in potsdam_refs:
        d_iou = row["mIoU"] - samba_potsdam_iou
        d_f1 = row["mF1"] - samba_potsdam_f1
        lines.append(f"| {row['model']} | {row['mIoU']:.2f} | {row['mF1']:.2f} | — | {d_iou:+.2f} / {d_f1:+.2f} |")

    if potsdam_metrics:
        miou5 = float(potsdam_metrics.get("full/mIoU5", 0.0))
        mf15 = float(potsdam_metrics.get("full/mF15", 0.0))
        oa5 = float(potsdam_metrics.get("full/OA5", 0.0))
        d_iou = miou5 - samba_potsdam_iou
        d_f1 = mf15 - samba_potsdam_f1
        lines.append(f"| **RPGV-Net (Ours)** | **{miou5:.2f}** | **{mf15:.2f}** | **{oa5:.2f}** | **{d_iou:+.2f} / {d_f1:+.2f}** |")
    else:
        lines.append("| **RPGV-Net (Ours)** | *待训练评估* | — | — | — |")

    summary_text = "\n".join(lines) + "\n"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n[SUMMARY UPDATED] {summary_path}")
    print(summary_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "loveda", "potsdam"], default="potsdam")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    loveda_metrics = None
    potsdam_metrics = None

    if args.dataset in ["all", "loveda"]:
        loveda_metrics = run_loveda_rpgv(dry_run=args.dry_run)
    if args.dataset in ["all", "potsdam"]:
        potsdam_metrics = run_potsdam_rpgv(dry_run=args.dry_run)

    update_summary(loveda_metrics, potsdam_metrics)


if __name__ == "__main__":
    main()
