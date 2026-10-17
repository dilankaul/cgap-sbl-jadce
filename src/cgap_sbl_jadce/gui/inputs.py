import hashlib
import json

from cgap_sbl_jadce.checkpoints import scientific_revision
from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.settings import tuning_settings, tuning_pairs, request_settings
from cgap_sbl_jadce.paths import task_dir, result_path, tuning_path


def input_signature(request, portable=False):
    cfg = load_config(request['config'])
    if request.get('kind') not in {'task', 'preview'}:
        return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    out = cfg['run']['output_dir']
    task = request.get('task')
    action = request.get('action', 'run')
    paths = []
    if task == 'activity':
        paths.append(result_path(out, 'kappa_calibration', 'summary.json'))
    if task in {'communication', 'spatial_dependence'}:
        paths += [result_path(out, 'activity', f'{split}_activity.npz') for split in ('tuning', 'evaluation')]
        paths.append(result_path(out, 'activity', 'summary.json'))
    if task in {'omp', 'cosamp', 'sbl', 'anc', 'gl', 'comparison', 'convergence'}:
        split = 'tuning' if action == 'tune' else 'evaluation'
        folder = task_dir(out, 'communication') / split
        count = cfg['run'][f'num_{split}_samples']
        for q in range(count):
            paths += [folder / f'{q:04d}' / name for name in ('channels.npz', 'pilots.npz', 'observations.npz')]
        if task in {'anc', 'gl'} or task == 'convergence' and set(request.get('methods', ['sbl','anc','gl'])) & {'anc', 'gl'}:
            paths += [task_dir(out, 'spatial_dependence') / split / f'{q:04d}.npz' for q in range(count)]
        if action == 'run':
            methods = [task] if task in {'sbl', 'anc', 'gl'} else request.get('methods', ['sbl', 'anc', 'gl']) if task == 'convergence' else []
            for method in methods:
                if method not in {'sbl', 'anc', 'gl'}:
                    continue
                if tuning_settings(cfg, method)['selection_mode'] == 'per_condition':
                    if task == 'convergence':
                        from cgap_sbl_jadce.settings import evaluation_settings
                        pairs = [tuple(request.get('condition') or (cfg['evaluation']['pilot_length'], evaluation_settings(cfg)['snr_db']))]
                    else:
                        pairs = request.get('conditions') or tuning_pairs(cfg, method)
                    for condition in pairs:
                        paths.append(tuning_path(cfg, method, 'selected.json', condition))
                else:
                    paths.append(tuning_path(cfg, method, 'selected.json'))
    if task == 'anc_grid_study':
        from cgap_sbl_jadce.experiments.anc_grid_study import grid_definition
        return hashlib.sha256(json.dumps([scientific_revision(), grid_definition(cfg)], sort_keys=True).encode()).hexdigest()
    if task == 'comparison':
        for method in request.get('methods', ['omp', 'cosamp', 'sbl', 'anc', 'gl']):
            paths += [result_path(out, method, name) for name in ('per_sample.csv', 'run_summary.json')]
    from cgap_sbl_jadce.experiments.estimators import ESTIMATORS, evaluation_conditions, tuning_conditions
    identities = []
    if task in ESTIMATORS or task == 'convergence':
        from cgap_sbl_jadce.communication_dataset import communication_condition_identity
        split = 'tuning' if action == 'tune' else 'evaluation'
        if task == 'convergence':
            from cgap_sbl_jadce.settings import evaluation_settings
            pairs = [tuple(request.get('condition') or (cfg['evaluation']['pilot_length'], evaluation_settings(cfg)['snr_db']))]
        else:
            pairs = tuning_conditions(cfg, request.get('conditions'), task) if action == 'tune' else list(dict.fromkeys(v[:2] for v in evaluation_conditions(cfg, request.get('conditions'), method=task)))
        identities = [communication_condition_identity(cfg, split, pair) for pair in pairs]
        comm = task_dir(out, 'communication')
        paths = [path for path in paths if not path.is_relative_to(comm)]
    if portable:
        from cgap_sbl_jadce.checkpoints import _archive_digest
        identity = []
        for path in paths:
            name = path.relative_to(out).as_posix()
            if path.exists():
                info = path.stat()
                digest = _archive_digest(str(path.resolve()), info.st_size, info.st_mtime_ns, info.st_ctime_ns).hex()
            else:
                digest = None
            identity.append((name, digest))
    else:
        identity = [(str(path), path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else (str(path), None, None)
                    for path in paths]
    return hashlib.sha256(json.dumps([scientific_revision(), request_settings(cfg, request), identities, identity]).encode()).hexdigest()
