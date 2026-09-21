#!/usr/bin/env python3
"""Summarize validation and test metrics across RPGV ablation experiments."""

import glob
import json
import os
import sys
from pathlib import Path


def get_best_val(ablation_dir):
    scalars_files = sorted(glob.glob(os.path.join(ablation_dir, '**', 'vis_data', 'scalars.json'), recursive=True))
    best_record = None
    best_iou = -1.0
    for sf in scalars_files:
        with open(sf, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    if 'binary/Foreground_IoU' in d and 'step' in d:
                        iou = float(d['binary/Foreground_IoU'])
                        if iou > best_iou:
                            best_iou = iou
                            best_record = d
                except Exception:
                    pass
    return best_record


def get_test_eval(ablation_dir):
    eval_files = sorted(glob.glob(os.path.join(ablation_dir, 'test_eval', '**', '*.json'), recursive=True))
    for ef in eval_files:
        try:
            with open(ef, 'r', encoding='utf-8') as f:
                d = json.load(f)
                if 'binary/Foreground_IoU' in d:
                    return d
        except Exception:
            pass
    return None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'work_dirs/rpgv_ablations'
    root_path = Path(root)
    if not root_path.is_dir():
        print(f"Directory {root} does not exist.")
        return

    exp_dirs = sorted([p for p in root_path.iterdir() if p.is_dir()])
    
    rows = []
    for ed in exp_dirs:
        name = ed.name
        stage3_dir = ed / 'stage3_joint'
        search_dir = str(stage3_dir) if stage3_dir.is_dir() else str(ed)
        val_rec = get_best_val(search_dir)
        test_rec = get_test_eval(search_dir)
        
        row = {
            'Ablation': name,
            'Val_Iter': val_rec.get('step', '-') if val_rec else '-',
            'Val_IoU': f"{val_rec['binary/Foreground_IoU']:.2f}%" if val_rec else '-',
            'Val_Dice': f"{val_rec['binary/Dice']:.2f}%" if val_rec else '-',
            'Val_HD95_m': f"{val_rec.get('binary/HD95_m', val_rec.get('binary/HD95_px', 0.0) * 0.5):.1f}m" if val_rec else '-',
            'Test_IoU': f"{test_rec['binary/Foreground_IoU']:.2f}%" if test_rec else '-',
            'Test_Dice': f"{test_rec['binary/Dice']:.2f}%" if test_rec else '-',
            'Test_HD95_m': f"{test_rec.get('binary/HD95_m', test_rec.get('binary/HD95_px', 0.0) * 0.5):.1f}m" if test_rec else '-',
        }
        rows.append(row)

    headers = ['Ablation', 'Val_Iter', 'Val_IoU', 'Val_Dice', 'Val_HD95_m', 'Test_IoU', 'Test_Dice', 'Test_HD95_m']
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) if rows else len(h) for h in headers}
    
    header_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"
    print(header_line)
    print(sep_line)
    for r in rows:
        print("| " + " | ".join(str(r[h]).ljust(col_widths[h]) for h in headers) + " |")


if __name__ == '__main__':
    main()
