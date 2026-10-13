"""Detect dependencies changed by another task while a worker is paused."""
from pathlib import Path
import hashlib
import json

from cgap_sbl_jadce.checkpoints import scientific_revision
from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.paths import task_dir, result_path, tuning_path


def input_signature(request):
    cfg = load_config(request['config'])
    out = cfg['run']['output_dir']
    task = request.get('task')
    action = request.get('action', 'run')
    paths = []
    if task == 'activity':
        paths.append(result_path(out, 'kappa_calibration', 'summary.json'))
    if task in {'communication', 'spatial_dependence'}:
        paths += [result_path(out, 'activity', f'{split}_activity.npz') for split in ('tuning', 'evaluation')]
        paths.append(result_path(out, 'activity', 'summary.json'))
    if task in {'omp', 'cosamp', 'sbl', 'anc', 'comparison', 'convergence'}:
        split = 'tuning' if action == 'tune' else 'evaluation'
        folder = task_dir(out, 'communication') / split
        count = cfg['run'][f'num_{split}_samples']
        for q in range(count):
            paths += [folder / f'{q:04d}' / name for name in ('channels.npz', 'pilots.npz', 'observations.npz')]
        if task == 'anc' or task == 'convergence' and 'anc' in request.get('methods', ['sbl','anc']):
            paths += [task_dir(out, 'spatial_dependence') / split / f'{q:04d}.npz' for q in range(count)]
        if action == 'run':
            methods = [task] if task in {'sbl', 'anc'} else request.get('methods', ['sbl', 'anc']) if task == 'convergence' else []
            for method in methods:
                if method not in {'sbl', 'anc'}:
                    continue
                if cfg['tuning'].get('selection_mode') == 'per_condition':
                    for condition in cfg.get('communication', {}).get('tuning_conditions', []):
                        paths.append(tuning_path(cfg, method, 'selected.json', condition))
                else:
                    paths.append(tuning_path(cfg, method, 'selected.json'))
    if task == 'comparison':
        for method in request.get('methods', ['omp', 'cosamp', 'sbl', 'anc']):
            paths += [result_path(out, method, name) for name in ('per_sample.csv', 'run_summary.json')]
    identity = [(str(path), path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else (str(path), None, None)
                for path in paths]
    return hashlib.sha256(json.dumps([scientific_revision(), hashlib.sha256(Path(request['config']).read_bytes()).hexdigest(), identity]).encode()).hexdigest()
