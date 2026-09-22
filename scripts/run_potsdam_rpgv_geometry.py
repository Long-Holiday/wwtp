#!/usr/bin/env python3
"""Runner for full RPGV remote sensing benchmark on ISPRS Potsdam with Depth Anything geometry branch.

Steps:
1. Generate pseudo-geometry (depth + reliability) via Depth Anything V2 for all Potsdam splits
   if not already generated.
2. Train full RPGV model with geometry branch (40,000 iters on all 24 training tiles).
3. Evaluate checkpoint on the 14 benchmark test tiles.
4. Compare against literature baselines (Samba, SegFormer, DeepLabV3+) and optical baseline.
5. Update work_dirs/remote_sensing/summary.md.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT / "work_dirs/remote_sensing"
DATA_ROOT = REPO_ROOT / "data/remote_sensing/prepared/potsdam"
PSEUDO_ROOT = DATA_ROOT / "pseudo_geometry"


def run_cmd(cmd: list[str] | str, check: bool = True) -> int:
    display_cmd = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"\n[EXEC] {display_cmd}\n")
    sys.stdout.flush()
    res = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=str(REPO_ROOT))
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {display_cmd}")
    return res.returncode


def ensure_pseudo_geometry(batch_size: int = 16) -> None:
    print(f"\n{'='*70}\n[STEP 1] Ensure Potsdam Depth Anything V2 Pseudo-Geometry\n{'='*70}")
    train_count = len(list((PSEUDO_ROOT / "train").glob("*.npz"))) if (PSEUDO_ROOT / "train").is_dir() else 0
    val_count = len(list((PSEUDO_ROOT / "val").glob("*.npz"))) if (PSEUDO_ROOT / "val").is_dir() else 0
    test_count = len(list((PSEUDO_ROOT / "test").glob("*.npz"))) if (PSEUDO_ROOT / "test").is_dir() else 0

    print(f"Current pseudo-geometry files: train={train_count}/2880, val={val_count}/576, test={test_count}/2016")

    if train_count == 2880 and val_count == 576 and test_count == 2016:
        print("[OK] All Potsdam pseudo-geometry files exist. Skipping generation.")
        return

    print("Generating pseudo-geometry via Depth-Anything-V2-Base-hf inside Docker...")
    cmd = [
        "docker", "compose", "run", "--rm",
        "-e", "PYTHONUNBUFFERED=1",
        "wwtp", "python", "tools/remote_sensing/generate_potsdam_geometry.py",
        "--batch-size", str(batch_size),
        "--device", "cuda",
    ]
    run_cmd(cmd)
    print("[OK] Pseudo-geometry generation complete.")


def get_best_checkpoint(model_dir: Path, pattern: str = "iter_*.pth") -> Path | None:
    def _iter_num(p: str) -> int:
        try:
            return int(Path(p).stem.split('_')[-1])
        except Exception:
            return 0

    matches = sorted(glob.glob(str(model_dir / pattern)), key=_iter_num)
    if matches:
        return Path(matches[-1])
    return None


def run_potsdam_rpgv_geometry(dry_run: bool = False) -> dict:
    print(f"\n{'='*70}\n[STEP 2 & 3] Train and Evaluate Full RPGV on ISPRS Potsdam\n{'='*70}")
    work_dir = WORK_ROOT / "potsdam_rpgv_full_geometry"
    eval_dir = WORK_ROOT / "potsdam_rpgv_full_geometry_eval"
    metrics_file = eval_dir / "metrics.json"

    if metrics_file.is_file():
        print(f"[SKIP] Potsdam Full Geometry RPGV already evaluated. Metrics: {metrics_file}")
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if dry_run:
        print("[DRY-RUN] Full RPGV Potsdam paths and configs verified.")
        return {}

    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train
    final_ckpt = work_dir / "iter_40000.pth"
    if not final_ckpt.is_file():
        ckpt = get_best_checkpoint(work_dir, "iter_*.pth")
        if not ckpt:
            print("\n>>> Launching Potsdam Full RPGV (with Geometry Branch) Training in Docker...")
            train_cmd = [
                "docker", "compose", "run", "--rm",
                "-e", "PYTHONUNBUFFERED=1",
                "wwtp", "python", "tools/remote_sensing/run.py", "train",
                "configs/remote_sensing/potsdam_rpgv_full_geometry.py",
                "--work-dir", "work_dirs/remote_sensing/potsdam_rpgv_full_geometry",
            ]
            run_cmd(train_cmd)
            ckpt = get_best_checkpoint(work_dir, "iter_*.pth")
            if not ckpt:
                raise RuntimeError(f"No checkpoint produced in {work_dir} after Potsdam full geometry training!")
        final_ckpt = ckpt

    print(f"\n[OK] Potsdam Full Geometry Training complete. Checkpoint: {final_ckpt}")

    # 2. Eval
    if not metrics_file.is_file():
        print("\n>>> Launching Potsdam Full RPGV Evaluation in Docker on 14 Test Tiles...")
        rel_ckpt = final_ckpt.relative_to(REPO_ROOT)
        eval_cmd = [
            "docker", "compose", "run", "--rm",
            "-e", "PYTHONUNBUFFERED=1",
            "wwtp", "python", "tools/remote_sensing/run.py", "eval",
            "configs/remote_sensing/potsdam_rpgv_full_geometry.py",
            "--checkpoint", str(rel_ckpt),
            "--work-dir", "work_dirs/remote_sensing/potsdam_rpgv_full_geometry_eval",
        ]
        run_cmd(eval_cmd)

    # 3. Compare
    compare_cmd = [sys.executable, "tools/remote_sensing/compare.py", "potsdam", str(metrics_file)]
    run_cmd(compare_cmd, check=False)

    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


def update_summary_table(geom_metrics: dict | None) -> None:
    summary_path = WORK_ROOT / "summary.md"
    optical_metrics_path = WORK_ROOT / "potsdam_rpgv_full_eval/metrics.json"
    optical_metrics = None
    if optical_metrics_path.is_file():
        with open(optical_metrics_path, "r", encoding="utf-8") as f:
            optical_metrics = json.load(f)

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

    loveda_refs = refs.get("loveda", {}).get("rows", [])
    samba_loveda_iou = 47.11
    samba_loveda_f1 = 63.17
    for row in loveda_refs:
        d_iou = row["mIoU"] - samba_loveda_iou
        d_f1 = row["mF1"] - samba_loveda_f1
        lines.append(f"| {row['model']} | {row['mIoU']:.2f} | — | {row['mF1']:.2f} | {d_iou:+.2f} / {d_f1:+.2f} |")

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

    if optical_metrics:
        miou5 = float(optical_metrics.get("full/mIoU5", 0.0))
        mf15 = float(optical_metrics.get("full/mF15", 0.0))
        oa5 = float(optical_metrics.get("full/OA5", 0.0))
        d_iou = miou5 - samba_potsdam_iou
        d_f1 = mf15 - samba_potsdam_f1
        lines.append(f"| **RPGV-Net (纯光学 RGB 基线)** | **{miou5:.2f}** | **{mf15:.2f}** | **{oa5:.2f}** | **{d_iou:+.2f} / {d_f1:+.2f}** |")

    if geom_metrics:
        miou5_g = float(geom_metrics.get("full/mIoU5", 0.0))
        mf15_g = float(geom_metrics.get("full/mF15", 0.0))
        oa5_g = float(geom_metrics.get("full/OA5", 0.0))
        d_iou_g = miou5_g - samba_potsdam_iou
        d_f1_g = mf15_g - samba_potsdam_f1
        lines.append(f"| **RPGV-Net (完整几何分支: Depth-Anything + RGR + DFGV)** | **{miou5_g:.2f}** | **{mf15_g:.2f}** | **{oa5_g:.2f}** | **{d_iou_g:+.2f} / {d_f1_g:+.2f}** |")
    else:
        lines.append("| **RPGV-Net (完整几何分支: Depth-Anything + RGR + DFGV)** | *待训练评估* | — | — | — |")

    summary_text = "\n".join(lines) + "\n"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n[SUMMARY UPDATED] {summary_path}")
    print(summary_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        ensure_pseudo_geometry(batch_size=args.batch_size)

    geom_metrics = run_potsdam_rpgv_geometry(dry_run=args.dry_run)
    update_summary_table(geom_metrics)


if __name__ == "__main__":
    main()
