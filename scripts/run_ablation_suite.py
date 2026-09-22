#!/usr/bin/env python3
"""Automated runner for RPGV-Net ablation experiments."""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_STAGE2_CKPT = "work_dirs/rpgv_staged/stage2_geometry/best_binary_Foreground_IoU_iter_10000.pth"
DEFAULT_WORK_ROOT = "work_dirs/rpgv_ablations"

PLAN_ABLATIONS = [
    "no_rgr",
    "no_frequency_validation",
    "no_global_context",
    "unweighted_fusion",
    "no_detail_refinement",
    "no_depth_rectification",
    "no_geometry_fusion",
]

# These variants remain explicitly selectable, but are outside the reduced
# default plan. See docs/model_design.md for the evidence and tradeoffs.
OPTIONAL_ABLATIONS = [
    "offline_reliability_only",
    "no_boundary_fusion",
    "no_region_fusion",
    "no_boundary_refinement",
    "no_shape_auxiliary",
    "no_geometry_dropout",
]
ALL_ABLATIONS = PLAN_ABLATIONS + OPTIONAL_ABLATIONS
PENDING_ABLATIONS = ["no_depth_rectification", "no_geometry_fusion"]


def run_cmd(cmd, check=True):
    print(f"\n[EXEC] {' '.join(cmd) if isinstance(cmd, list) else cmd}\n")
    sys.stdout.flush()
    res = subprocess.run(cmd, shell=isinstance(cmd, str))
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd}")
    return res.returncode


def get_best_ckpt(ablation_dir):
    ckpts = sorted(glob.glob(os.path.join(ablation_dir, "best_binary_Foreground_IoU_iter_*.pth")))
    if ckpts:
        return ckpts[-1]
    ckpts = sorted(glob.glob(os.path.join(ablation_dir, "iter_*.pth")))
    if ckpts:
        return ckpts[-1]
    return None


def get_best_val(ablation_dir):
    scalars_files = sorted(glob.glob(os.path.join(ablation_dir, "**", "vis_data", "scalars.json"), recursive=True))
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


def get_test_eval(ablation_dir):
    eval_files = sorted(glob.glob(os.path.join(ablation_dir, "test_eval", "**", "*.json"), recursive=True))
    for ef in reversed(eval_files):
        try:
            with open(ef, "r", encoding="utf-8") as f:
                d = json.load(f)
                if "binary/Foreground_IoU" in d:
                    return d
        except Exception:
            pass
    return None


def update_summary(work_root):
    root_path = Path(work_root)
    if not root_path.is_dir():
        return

    exp_dirs = sorted([p for p in root_path.iterdir() if p.is_dir()])
    rows = []
    for ed in exp_dirs:
        name = ed.name
        stage3_dir = ed / "stage3_joint"
        search_dir = str(stage3_dir) if stage3_dir.is_dir() else str(ed)
        val_rec = get_best_val(search_dir)
        test_rec = get_test_eval(search_dir)
        
        row = {
            "Ablation": name,
            "Val_Iter": str(val_rec.get("step", "-")) if val_rec else "-",
            "Val_IoU": f"{val_rec['binary/Foreground_IoU']:.2f}%" if val_rec else "-",
            "Val_Dice": f"{val_rec['binary/Dice']:.2f}%" if val_rec else "-",
            "Val_HD95_m": f"{val_rec.get('binary/HD95_m', val_rec.get('binary/HD95_px', 0.0) * 0.5):.1f}m" if val_rec else "-",
            "Test_IoU": f"{test_rec['binary/Foreground_IoU']:.2f}%" if test_rec else "-",
            "Test_Dice": f"{test_rec['binary/Dice']:.2f}%" if test_rec else "-",
            "Test_HD95_m": f"{test_rec.get('binary/HD95_m', test_rec.get('binary/HD95_px', 0.0) * 0.5):.1f}m" if test_rec else "-",
        }
        rows.append(row)

    headers = ["Ablation", "Val_Iter", "Val_IoU", "Val_Dice", "Val_HD95_m", "Test_IoU", "Test_Dice", "Test_HD95_m"]
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) if rows else len(h) for h in headers}
    
    header_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"
    
    lines = [
        "# RPGV-Net 消融实验全维度综合对比表\n",
        f"*更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
        header_line,
        sep_line
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[h]).ljust(col_widths[h]) for h in headers) + " |")
    
    summary_text = "\n".join(lines) + "\n"
    summary_path = root_path / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("\n" + summary_text)


def run_ablation(ablation, work_root=DEFAULT_WORK_ROOT, stage2_ckpt=DEFAULT_STAGE2_CKPT):
    config = f"configs/ablations/rpgv_{ablation}.py"
    if not os.path.isfile(config):
        raise FileNotFoundError(f"Config {config} does not exist!")

    ablation_dir = os.path.join(work_root, ablation, "stage3_joint")
    os.makedirs(ablation_dir, exist_ok=True)
    test_eval_dir = os.path.join(ablation_dir, "test_eval")
    training_done = Path(ablation_dir) / ".training_completed"

    print(f"\n{'='*70}\n[START] Ablation: {ablation}\nConfig: {config}\nWorkDir: {ablation_dir}\n{'='*70}")

    best_ckpt = get_best_ckpt(ablation_dir)
    test_res = get_test_eval(ablation_dir)

    if best_ckpt and test_res:
        print(f"[SKIP] Ablation {ablation} already finished training & evaluation.")
        print(f"       Best Checkpoint: {best_ckpt}")
        print(f"       Test IoU: {test_res.get('binary/Foreground_IoU', 0.0):.2f}%")
        update_summary(work_root)
        return

    # A best checkpoint may appear at the first validation while training is
    # still active. Only this runner's completion marker permits evaluation.
    if not training_done.is_file():
        if any(Path(ablation_dir).iterdir()):
            raise RuntimeError(
                f"Incomplete or externally managed run in {ablation_dir}; "
                "refusing to test a checkpoint from active training. "
                "Use a fresh --work-root or finish this run first.")
        print(f"\n>>> [Stage 1/2] Launching Training for {ablation}...")
        train_cmd = [
            "docker", "compose", "run", "--rm",
            "-e", "PYTHONUNBUFFERED=1",
            "-e", f"RPGV_STAGE2_CHECKPOINT={stage2_ckpt}",
            "wwtp", "python", "tools/train.py", config,
            "--work-dir", ablation_dir,
            "--cfg-options",
            "train_dataloader.batch_size=1",
            "train_dataloader.num_workers=4",
            "val_dataloader.num_workers=4",
            "optim_wrapper.accumulative_counts=8",
            "train_cfg.max_iters=16000",
            "train_cfg.val_interval=1000",
            "param_scheduler.0.end=750",
            "param_scheduler.1.begin=750",
            "param_scheduler.1.end=16000"
        ]
        run_cmd(train_cmd)
        best_ckpt = get_best_ckpt(ablation_dir)
        if not best_ckpt:
            raise RuntimeError(f"No checkpoint produced in {ablation_dir} after training!")
        training_done.write_text(
            f"config={config}\nstage2_checkpoint={stage2_ckpt}\n",
            encoding="utf-8")

    if not best_ckpt:
        raise RuntimeError(f"No checkpoint produced in {ablation_dir} after training!")

    print(f"\n[OK] Training completed. Best checkpoint: {best_ckpt}")

    # Step 2: Test set evaluation
    if not test_res:
        print(f"\n>>> [Stage 2/2] Launching Test Set Evaluation for {ablation}...")
        test_cmd = [
            "docker", "compose", "run", "--rm",
            "-e", "PYTHONUNBUFFERED=1",
            "wwtp", "python", "tools/test.py", config, best_ckpt,
            "--work-dir", test_eval_dir
        ]
        run_cmd(test_cmd)

    update_summary(work_root)
    print(f"\n[DONE] Ablation {ablation} completed successfully!\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", choices=ALL_ABLATIONS + ["pending", "all"],
                        default="pending",
                        help="A single variant, the two pending variants, or all seven planned variants")
    parser.add_argument("--work-root", default=DEFAULT_WORK_ROOT,
                        help="Root work directory for ablations")
    parser.add_argument("--stage2-ckpt", default=DEFAULT_STAGE2_CKPT,
                        help="Shared Stage 2 checkpoint")
    args = parser.parse_args()

    # Preserve the existing full and RGB-only references in the summary.
    os.makedirs(os.path.join(args.work_root, "full"), exist_ok=True)
    os.makedirs(os.path.join(args.work_root, "rgb_only"), exist_ok=True)
    full_link = os.path.join(args.work_root, "full", "stage3_joint")
    rgb_link = os.path.join(args.work_root, "rgb_only", "stage3_joint")
    if not os.path.exists(full_link):
        os.symlink("../../rpgv_staged/stage3_joint", full_link)
    if not os.path.exists(rgb_link):
        os.symlink("../../rpgv_staged/stage1_rgb", rgb_link)

    if args.ablation == "pending":
        targets = PENDING_ABLATIONS
    elif args.ablation == "all":
        targets = PLAN_ABLATIONS
    else:
        targets = [args.ablation]
    for ab in targets:
        run_ablation(ab, work_root=args.work_root, stage2_ckpt=args.stage2_ckpt)


if __name__ == "__main__":
    main()
