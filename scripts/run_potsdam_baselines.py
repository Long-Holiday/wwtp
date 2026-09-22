#!/usr/bin/env python3
"""Sequential runner for remote sensing baseline models on ISPRS Potsdam benchmark.

Models trained and evaluated under the exact same standard protocol:
- DeepLabV3+ (ResNet-50)
- SegFormer (MiT-B2)
- UNetFormer (ResNet-18)
- Mask2Former (Swin-T)

Protocol:
- All 24 historical training tiles (3,456 patches of 512x512)
- 32,000 iterations with batch size 4
- No test leakage during training
- Final evaluation on 14 historical test tiles (2,016 patches)
- 5-class metrics (excluding clutter): mIoU, mF1, OA, plus eroded-3 boundary protocol.
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
SUMMARY_PATH = WORK_ROOT / "summary.md"
BASELINE_MAX_ITERS = 32000

BASELINE_CONFIGS = {
    "deeplabv3plus": {
        "name": "DeepLabV3+ (ResNet-50)",
        "config": "configs/remote_sensing/potsdam_deeplabv3plus.py",
        "work_dir": "work_dirs/remote_sensing/potsdam_deeplabv3plus",
        "eval_dir": "work_dirs/remote_sensing/potsdam_deeplabv3plus_eval",
    },
    "segformer": {
        "name": "SegFormer (MiT-B2)",
        "config": "configs/remote_sensing/potsdam_segformer_full.py",
        "work_dir": "work_dirs/remote_sensing/potsdam_segformer_full",
        "eval_dir": "work_dirs/remote_sensing/potsdam_segformer_full_eval",
    },
    "unetformer": {
        "name": "UNetFormer (ResNet-18)",
        "config": "configs/remote_sensing/potsdam_unetformer.py",
        "work_dir": "work_dirs/remote_sensing/potsdam_unetformer",
        "eval_dir": "work_dirs/remote_sensing/potsdam_unetformer_eval",
    },
    "mask2former": {
        "name": "Mask2Former (Swin-T)",
        "config": "configs/remote_sensing/potsdam_mask2former.py",
        "work_dir": "work_dirs/remote_sensing/potsdam_mask2former",
        "eval_dir": "work_dirs/remote_sensing/potsdam_mask2former_eval",
    },
}



def run_cmd(cmd: list[str] | str, check: bool = True) -> int:
    display_cmd = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"\n[EXEC] {display_cmd}\n")
    sys.stdout.flush()
    res = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=str(REPO_ROOT))
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {display_cmd}")
    return res.returncode


def get_latest_checkpoint(model_dir: Path, pattern: str = "iter_*.pth") -> Path | None:
    def _iter_num(p: str) -> int:
        try:
            return int(Path(p).stem.split("_")[-1])
        except Exception:
            return 0

    matches = sorted(glob.glob(str(model_dir / pattern)), key=_iter_num)
    if matches:
        return Path(matches[-1])
    return None


def train_model(model_key: str, info: dict, dry_run: bool = False) -> Path:
    config_path = REPO_ROOT / info["config"]
    work_dir = REPO_ROOT / info["work_dir"]
    work_dir.mkdir(parents=True, exist_ok=True)

    final_ckpt = work_dir / f"iter_{BASELINE_MAX_ITERS}.pth"
    if final_ckpt.is_file():
        print(f"[{model_key}] Final checkpoint already exists: {final_ckpt}")
        return final_ckpt

    latest_ckpt = get_latest_checkpoint(work_dir)
    if dry_run:
        print(f"[DRY-RUN] Would train {model_key} using {config_path}")
        return final_ckpt

    train_cmd = [
        "docker", "compose", "run", "--rm",
        "-e", "PYTHONUNBUFFERED=1",
        "wwtp", "python", "tools/remote_sensing/run.py", "train",
        str(info["config"]),
        "--work-dir", str(info["work_dir"]),
    ]
    if latest_ckpt:
        print(f"[{model_key}] Found existing checkpoint {latest_ckpt}, resuming training...")
        train_cmd.append("--resume")
    else:
        print(f"[{model_key}] Launching fresh {BASELINE_MAX_ITERS:,} iters training...")

    run_cmd(train_cmd)

    if not final_ckpt.is_file():
        raise RuntimeError(f"[{model_key}] Expected final checkpoint missing: {final_ckpt}")
    return final_ckpt


def evaluate_model(model_key: str, info: dict, checkpoint: Path, dry_run: bool = False) -> dict:
    eval_dir = REPO_ROOT / info["eval_dir"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = eval_dir / "metrics.json"

    if metrics_file.is_file():
        print(f"[{model_key}] Evaluation already completed: {metrics_file}")
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if dry_run:
        print(f"[DRY-RUN] Would evaluate {model_key} on 14 test tiles")
        return {}

    rel_ckpt = checkpoint.relative_to(REPO_ROOT) if checkpoint.is_absolute() else checkpoint
    eval_cmd = [
        "docker", "compose", "run", "--rm",
        "-e", "PYTHONUNBUFFERED=1",
        "wwtp", "python", "tools/remote_sensing/run.py", "eval",
        str(info["config"]),
        "--checkpoint", str(rel_ckpt),
        "--work-dir", str(info["eval_dir"]),
    ]
    run_cmd(eval_cmd)

    if not metrics_file.is_file():
        raise RuntimeError(f"[{model_key}] Metrics file not generated at {metrics_file}")

    # Run benchmark comparison script
    compare_cmd = [sys.executable, "tools/remote_sensing/compare.py", "potsdam", str(metrics_file)]
    run_cmd(compare_cmd, check=False)

    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


def update_summary():
    """Consolidate all trained and evaluated models into summary.md."""
    refs_path = REPO_ROOT / "tools/remote_sensing/references.json"
    refs = {}
    if refs_path.is_file():
        with open(refs_path, "r", encoding="utf-8") as f:
            refs = json.load(f)

    lines = [
        "# 通用遥感语义分割基准测试与对比模型汇总表",
        f"\n*最后更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "## 1. 实验协议与基准说明",
        "- **数据集**: ISPRS Potsdam (24张历史训练瓦片 = 3,456个512x512切片; 14张独立测试瓦片 = 2,016个512x512切片)",
        "- **训练协议**: RPGV 40,000 steps；本地 Baseline 32,000 steps；batch_size=4；最终训练不使用留出验证集，在历史 14 张测试瓦片上独立评估",
        "- **评估指标**: 5类 mIoU (%), 5类 mF1 (%), 5类 OA (%) [排除 Clutter 背景类], 以及 3-pixel 边界腐蚀协议 (Eroded-3)\n",
        "## 2. ISPRS Potsdam 数据集评测对比总表",
        "| 序号 | 模型 / 方法 | 类别 / 架构 | 5类 mIoU (%) | 5类 mF1 (%) | 5类 OA (%) | Eroded-3 mIoU (%) | 与 Samba 差值 (mIoU pp) | 状态 |",
        "|---|---|---|---:|---:|---:|---:|---:|:---:|",
    ]

    samba_potsdam_iou = 82.29
    samba_potsdam_f1 = 90.15

    # Literature references
    potsdam_refs = refs.get("potsdam", {}).get("rows", [])
    row_idx = 1
    for row in potsdam_refs:
        d_iou = row["mIoU"] - samba_potsdam_iou
        lines.append(
            f"| {row_idx} | {row['model']} | 文献发表基线 | {row['mIoU']:.2f} | {row['mF1']:.2f} | — | — | {d_iou:+.2f} | 文献发布 |"
        )
        row_idx += 1

    # Our models mapping
    eval_targets = [
        ("RPGV-Net (纯光学 RGB 基线)", "work_dirs/remote_sensing/potsdam_rpgv_full_eval/metrics.json", "Ours (RGB Only)"),
        ("RPGV-Net (完整几何分支: Depth-Anything + RGR + DFGV)", "work_dirs/remote_sensing/potsdam_rpgv_full_geometry_eval/metrics.json", "Ours (Full Multi-Modal)"),
        ("DeepLabV3+ (ResNet-50)", "work_dirs/remote_sensing/potsdam_deeplabv3plus_eval/metrics.json", "CNN Baseline"),
        ("SegFormer (MiT-B2)", "work_dirs/remote_sensing/potsdam_segformer_full_eval/metrics.json", "Transformer Baseline"),
        ("UNetFormer (ResNet-18)", "work_dirs/remote_sensing/potsdam_unetformer_eval/metrics.json", "Remote Sensing SOTA"),
        ("Mask2Former (Swin-T)", "work_dirs/remote_sensing/potsdam_mask2former_eval/metrics.json", "Mask Attention SOTA"),
    ]

    for name, mpath, arch in eval_targets:
        full_mpath = REPO_ROOT / mpath
        if full_mpath.is_file():
            try:
                with open(full_mpath, "r", encoding="utf-8") as f:
                    m = json.load(f)
                miou5 = float(m.get("full/mIoU5", 0.0))
                mf15 = float(m.get("full/mF15", 0.0))
                oa5 = float(m.get("full/OA5", 0.0))
                eroded_iou = float(m.get("eroded3/mIoU5", 0.0))
                d_iou = miou5 - samba_potsdam_iou
                lines.append(
                    f"| {row_idx} | **{name}** | {arch} | **{miou5:.2f}** | **{mf15:.2f}** | **{oa5:.2f}** | **{eroded_iou:.2f}** | **{d_iou:+.2f}** | ✅ 完成 |"
                )
            except Exception as e:
                lines.append(f"| {row_idx} | **{name}** | {arch} | 读取错误 | — | — | — | — | ❌ 异常 |")
        else:
            lines.append(f"| {row_idx} | **{name}** | {arch} | *待训练评估* | — | — | — | — | ⏳ 队列中 |")
        row_idx += 1

    summary_text = "\n".join(lines) + "\n"
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n[SUMMARY UPDATED] {SUMMARY_PATH}\n")
    print(summary_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deeplabv3plus", "segformer", "unetformer", "mask2former"],
        choices=list(BASELINE_CONFIGS.keys()),
        help="Models to train and evaluate",
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only evaluate")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = parser.parse_args()

    print("=" * 80)
    print(f"ISPRS Potsdam Baseline Runner: {len(args.models)} models requested: {', '.join(args.models)}")
    print("=" * 80)

    for idx, model_key in enumerate(args.models, 1):
        info = BASELINE_CONFIGS[model_key]
        print(f"\n[{idx}/{len(args.models)}] Processing {info['name']}...")

        # 1. Train
        if not args.skip_train:
            ckpt = train_model(model_key, info, dry_run=args.dry_run)
        else:
            ckpt = REPO_ROOT / info["work_dir"] / f"iter_{BASELINE_MAX_ITERS}.pth"
            if not ckpt.is_file():
                print(f"[{model_key}] Warning: Final checkpoint missing: {ckpt}; skipping eval.")
                continue

        # 2. Evaluate
        if not args.skip_eval and ckpt:
            evaluate_model(model_key, info, ckpt, dry_run=args.dry_run)

        # 3. Update summary after each model
        if not args.dry_run:
            update_summary()

    print("\n" + "=" * 80)
    print("All requested baselines processed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
