"""Build the complete 1m benchmarking and ablation audit comparing SegNeXt, RPGV-v5, and RPGV-v5.1."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

VAL_KEYS = (
    'binary/Foreground_IoU', 'binary/Dice', 'binary/Precision',
    'binary/Recall', 'binary/Boundary_F1', 'binary/Boundary_Precision',
    'binary/Boundary_Recall', 'binary/HD95_px',
    'topology/images', 'topology/negative_images',
    'topology/negative_images_with_fp', 'topology/gt_components',
    'topology/pred_components', 'topology/pred_small_components',
    'topology/pred_holes', 'topology/pred_hole_pixels',
    'topology/detached_fp_components', 'topology/detached_fp_pixels',
    'topology/fp_pixels', 'topology/detached_fp_fraction',
)

def extract_run(name, timestamp, max_iters):
    run_dir = ROOT / 'work_dirs' / name / timestamp
    scalar_path = run_dir / 'vis_data/scalars.json'
    rows = [json.loads(line) for line in scalar_path.read_text().splitlines() if line.strip()]
    validation = [row for row in rows if 'binary/Foreground_IoU' in row]
    training = [row for row in rows if 'loss' in row]
    best = max(validation, key=lambda r: r['binary/Foreground_IoU'])
    final = validation[-1]
    
    return {
        'name': name,
        'timestamp': timestamp,
        'max_iters': max_iters,
        'val_checkpoints_count': len(validation),
        'best': {k: best.get(k) for k in VAL_KEYS if k in best} | {'step': best.get('step', best.get('iter'))},
        'final': {k: final.get(k) for k in VAL_KEYS if k in final} | {'step': final.get('step', final.get('iter'))},
        'curve': [
            {
                'step': r.get('step', r.get('iter')),
                'iou': r.get('binary/Foreground_IoU'),
                'dice': r.get('binary/Dice'),
                'precision': r.get('binary/Precision'),
                'recall': r.get('binary/Recall'),
                'bf1': r.get('binary/Boundary_F1'),
                'hd95_m': r.get('binary/HD95_m', r.get('binary/HD95_px')),
                'negative_images_with_fp': r.get('topology/negative_images_with_fp'),
                'pred_holes': r.get('topology/pred_holes'),
                'pred_small_components': r.get('topology/pred_small_components'),
            }
            for r in validation
        ]
    }

def main():
    segnext = extract_run('segnext_1m', '20260928_042114', 20000)
    v5 = extract_run('rpgv_v5_1m', '20260928_004513', 20000)
    v51 = extract_run('rpgv_v51_1m', '20260928_075523', 6000)
    
    diag_summary = json.loads((ROOT / 'docs/analysis/rpgv_v51_val_diagnostic_summary.json').read_text())
    
    audit = {
        'schema': 2,
        'task': 'wwtp_1m_semantic_segmentation',
        'models': {
            'segnext_mscan_s': segnext,
            'rpgv_v5': v5,
            'rpgv_v51': v51,
        },
        'v5_module_ablations': diag_summary,
    }
    
    out_path = ROOT / 'docs/analysis/rpgv_v51_comprehensive_audit.json'
    out_path.write_text(json.dumps(audit, indent=2))
    print(f'Wrote comprehensive audit to {out_path}')

if __name__ == '__main__':
    main()
