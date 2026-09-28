"""Rebuild the RPGV v5 evidence inventory from local experiment artifacts.

Run inside the existing project Docker image. This script performs no inference or
training and only writes the audit JSON named by --output.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('binary/Foreground_IoU', 'binary/Boundary_F1',
           'binary/Precision', 'binary/Recall', 'binary/HD95_px')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def curve(path):
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if 'binary/Foreground_IoU' in row]
    if not rows:
        return {'source': str(path.relative_to(ROOT)), 'evaluations': 0}
    best = max(rows, key=lambda row: row['binary/Foreground_IoU'])
    return {
        'source': str(path.relative_to(ROOT)), 'evaluations': len(rows),
        'best_step': best['step'], 'last_step': rows[-1]['step'],
        'best_metrics': {key: best.get(key) for key in METRICS},
        'last_metrics': {key: rows[-1].get(key) for key in METRICS},
    }


def metrics_file(path):
    doc = read(path)
    if doc is None:
        return None
    record = doc.get('metrics', doc)
    return {'source': str(path.relative_to(ROOT)),
            'identity': doc.get('identity'),
            'metrics': {key: record.get(key) for key in METRICS}}


def stage(name, relative_dir, test=None):
    directory = ROOT / relative_dir
    curves = sorted(directory.glob('*/vis_data/scalars.json'))
    # Timestamped training directories can contain several restart segments.
    result = {'name': name, 'directory': relative_dir,
              'curves': [curve(path) for path in curves],
              'completed': None, 'test': None}
    completed = directory / 'completed.json'
    if completed.is_file():
        doc = read(completed)
        result['completed'] = {
            'source': str(completed.relative_to(ROOT)),
            'budget': doc.get('budget'), 'last_val_iter': doc.get('last_val_iter'),
            'stopped_early': doc.get('stopped_early'),
            'checkpoint': doc.get('checkpoint'),
            'checkpoint_sha256': doc.get('checkpoint_sha256'),
            'initialization': doc.get('initialization'),
            'selected_iter': doc.get('validation', {}).get('selected_iter'),
            'metrics': {key: doc.get('validation', {}).get(key) for key in METRICS},
        }
    if test:
        result['test'] = metrics_file(ROOT / test)
    return result


def quick_screen(relative_dir):
    directory = ROOT / relative_dir
    out = {'directory': relative_dir, 'stages': {}}
    for stage_name in ('zero_shot', 'finetune_eval'):
        entries = {}
        for path in sorted((directory / stage_name).glob('*/metrics.json')):
            entries[path.parent.name] = metrics_file(path)
        out['stages'][stage_name] = entries
    identity = read(directory / 'finetune/full/identity.json')
    out['finetune_protocol'] = identity
    manifest = read(directory / 'mini_val.json')
    out['mini_val_count'] = len(manifest.get('samples', [])) if manifest else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='docs/analysis/rpgv_v5_experiment_audit.json')
    args = parser.parse_args()
    staged = [
        ('v1_stage1', 'work_dirs/rpgv_staged/stage1_rgb',
         'work_dirs/rpgv_staged/stage1_rgb/test_eval/test_results.json'),
        ('v1_stage2', 'work_dirs/rpgv_staged/stage2_geometry', None),
        ('v1_stage3', 'work_dirs/rpgv_staged/stage3_joint',
         'work_dirs/rpgv_staged/stage3_joint/test_eval/20260920_004330/20260920_004330.json'),
        ('v1_stage3_restart_30k', 'work_dirs/rpgv_stage3_restart_30k',
         'work_dirs/rpgv_stage3_restart_30k/test_eval/20260921_130631.json'),
        ('v1_no_depth_rectification', 'work_dirs/rpgv_ablations/no_depth_rectification/stage3_joint',
         'work_dirs/rpgv_ablations/no_depth_rectification/stage3_joint/test_eval/20260921_094015.json'),
        ('v1_no_detail_refinement', 'work_dirs/rpgv_ablations/no_detail_refinement/stage3_joint',
         'work_dirs/rpgv_ablations/no_detail_refinement/stage3_joint/test_eval/20260921_081003.json'),
        ('v1_no_frequency_validation', 'work_dirs/rpgv_ablations/no_frequency_validation/stage3_joint',
         'work_dirs/rpgv_ablations/no_frequency_validation/stage3_joint/test_eval/20260921_030306.json'),
        ('v1_no_geometry_fusion', 'work_dirs/rpgv_ablations/no_geometry_fusion/stage3_joint',
         'work_dirs/rpgv_ablations/no_geometry_fusion/stage3_joint/test_eval/20260921_110233.json'),
        ('v1_no_global_context', 'work_dirs/rpgv_ablations/no_global_context/stage3_joint',
         'work_dirs/rpgv_ablations/no_global_context/stage3_joint/test_eval/20260921_052818.json'),
        ('v1_no_rgr', 'work_dirs/rpgv_ablations/no_rgr/stage3_joint',
         'work_dirs/rpgv_ablations/no_rgr/stage3_joint/test_eval/20260921_013859.json'),
        ('v1_unweighted_fusion', 'work_dirs/rpgv_ablations/unweighted_fusion/stage3_joint',
         'work_dirs/rpgv_ablations/unweighted_fusion/stage3_joint/test_eval/20260921_065229.json'),
        ('v2_stage1', 'work_dirs/rpgv_v2_paper/full/seed_42/full/stage1_rgb', None),
        ('v2_stage2', 'work_dirs/rpgv_v2_paper/full/seed_42/full/stage2_geometry', None),
        ('v2_stage3', 'work_dirs/rpgv_v2_paper/full/seed_42/full/stage3_joint',
         'work_dirs/rpgv_v2_paper/full/seed_42/full/evaluation_test/metrics.json'),
        ('v2_joint_fixed', 'work_dirs/rpgv_v2_joint_fixed', None),
        ('v2_no_rgr_stage1_partial', 'work_dirs/rpgv_v2_paper/full/seed_42/no_rgr/stage1_rgb', None),
        ('v2_no_rgr_stage2', 'work_dirs/rpgv_v2_paper_reuse/paper_reuse/seed_42/no_rgr/stage2_geometry', None),
        ('v2_no_rgr_stage3', 'work_dirs/rpgv_v2_paper_reuse/paper_reuse/seed_42/no_rgr/stage3_joint', None),
        ('v2_no_frequency_stage3', 'work_dirs/rpgv_v2_paper_reuse/paper_reuse/seed_42/no_frequency_validation/stage3_joint', None),
        ('v2_unweighted_stage3', 'work_dirs/rpgv_v2_paper_reuse/paper_reuse/seed_42/unweighted_fusion/stage3_joint', None),
        ('v2_single_progressive_base', 'work_dirs/rpgv_v2_single_stage_fixed/single_stage/seed_42/progressive_base/joint', None),
        ('v2_single_progressive_rgr', 'work_dirs/rpgv_v2_single_stage_fixed/single_stage/seed_42/progressive_rgr/joint', None),
        ('v2_single_progressive_dfgv', 'work_dirs/rpgv_v2_single_stage_fixed/single_stage/seed_42/progressive_dfgv/joint', None),
        ('v4', 'work_dirs/rpgv_v4', 'work_dirs/rpgv_v4/test_eval/20260924_091541.json'),
    ]
    baselines = {}
    for path in sorted((ROOT / 'work_dirs').glob('*/test_results/*/*.json')):
        if 'vis_data' not in path.parts:
            baselines.setdefault(path.parts[-4], []).append(metrics_file(path))
    for path in sorted((ROOT / 'work_dirs/edge_baselines').glob('*/test_eval/*.json')):
        baselines.setdefault(path.parts[-3], []).append(metrics_file(path))
    v3_curves = sorted((ROOT / 'work_dirs').glob('rpgv_v3*/**/scalars.json'))
    result = {
        'schema': 1,
        'metric_units': 'percent except HD95_px',
        'selection': 'validation foreground IoU; independent test results where present',
        'stages': {name: stage(name, directory, test) for name, directory, test in staged},
        'quick_screens': [quick_screen('work_dirs/rpgv_v2_quick_screen'),
                          quick_screen('work_dirs/rpgv_v2_quick_screen_expanded')],
        'baselines': baselines,
        'v3_training_artifacts_found': bool(v3_curves),
        'v3_training_curves': [str(path.relative_to(ROOT)) for path in v3_curves],
        'v3_initialization': 'work_dirs/rpgv_v3_initialization/init.json',
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(f'Wrote {output}: {len(result["stages"])} stages, '
          f'{sum(len(s["stages"]["finetune_eval"]) for s in result["quick_screens"])} quick fine-tunes')


if __name__ == '__main__':
    main()
