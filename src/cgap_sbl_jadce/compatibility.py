"""Reversible compatibility, based on relevant values and recorded input identities."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from functools import lru_cache

from cgap_sbl_jadce.settings import calibration_settings, solver_settings
from cgap_sbl_jadce.paths import result_path


class IncompatibleResults(ValueError):
    pass


def differences(saved, current, prefix=''):
    if isinstance(saved, dict) and isinstance(current, dict):
        return [field for key in sorted(set(saved) | set(current)) for field in
                differences(saved.get(key), current.get(key), f'{prefix}.{key}' if prefix else key)]
    return [] if saved == current else [prefix]


def require_match(saved, current, label):
    changed = differences(saved, current)
    if changed:
        raise IncompatibleResults(f'{label} incompatible: changed {", ".join(changed)}. Restore matching settings or rerun.')


def preparation_settings(cfg, task, split=None):
    system, activity = cfg['system'], cfg['activity']
    values = {'N': system['N'], 'R': system['R'], 'V': activity['V'], 'D': activity['D']}
    if task == 'kappa_calibration':
        cal = calibration_settings(cfg)
        values.update(S=system['S'], **cal, seed=cfg['seeds']['kappa_calibration'],
                      validation_seed=cfg['seeds']['kappa_validation'])
        values['calibration_validation_samples'] = cal.get('calibration_validation_samples', cal['calibration_samples'])
    else:
        from cgap_sbl_jadce.config import resolve_activity_kappa
        values.update(kappa=resolve_activity_kappa(cfg)[0], exact_sparsity=activity.get('exact_sparsity', False),
                      S=system['S'], samples=cfg['run'][f'num_{split}_samples'],
                      seed=cfg['seeds'][f'activity_{split}'])
    files = ['models/activity.py']
    if task == 'kappa_calibration':
        files.append('experiments/calibration.py')
    values['code'] = _source_digest(files)
    return values


def validate_preparation(cfg, task, split=None):
    path = result_path(cfg['run']['output_dir'], task, 'summary.json')
    if not path.exists():
        raise FileNotFoundError(f'Run {task.replace("_", " ")} first.')
    summary = json.loads(path.read_text())
    if summary.get('compatibility'):
        saved = summary['compatibility'][split] if split else summary['compatibility']
        require_match(saved, preparation_settings(cfg, task, split), task.replace('_', ' ').capitalize())
    else:
        # Older runs retain their recorded configuration; never write invented provenance.
        manifest = path.parent / 'manifest.json'
        if manifest.exists():
            old = json.loads(manifest.read_text()).get('configuration')
            if old:
                if task == 'kappa_calibration':
                    saved, current = preparation_settings(old, task), preparation_settings(cfg, task)
                    saved.pop('code', None); current.pop('code', None)
                    require_match(saved, current, 'Kappa calibration')
                else:
                    current = preparation_settings(cfg, task, split)
                    saved = {key: old['system'][key] for key in ('N', 'R', 'S')}
                    saved.update({key: old['activity'][key] for key in ('V', 'D', 'kappa', 'exact_sparsity') if key in old['activity']})
                    saved.update(seed=old['seeds'][f'activity_{split}'], samples=old['run'][f'num_{split}_samples'])
                    require_match(saved, {key: current[key] for key in saved}, 'Activity')
        # Minimal legacy fixtures/archives may have only a subset of model metadata.
        if task == 'activity':
            current = preparation_settings(cfg, task, split) if manifest.exists() else {
                **{key: cfg['system'][key] for key in ('N', 'R')}, 'V': cfg['activity']['V'],
                'D': cfg['activity']['D'], 'target_mean_S': cfg['system']['S']}
            known = {key: summary[key] for key in current if key in summary}
            require_match(known, {key: current[key] for key in known}, 'Activity')
    if task == 'activity':
        archive = result_path(cfg['run']['output_dir'], 'activity', f'{split}_activity.npz')
        if not archive.exists():
            raise FileNotFoundError(f'Generate {split} activity first.')
        # Cheap archive shape inspection avoids deserializing all samples on every UI render.
        stat = archive.stat()
        if _activity_shape(str(archive.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (cfg['run'][f'num_{split}_samples'], cfg['system']['N']):
            raise IncompatibleResults(f'{split.capitalize()} activity incompatible: changed samples or N.')
    return summary


@lru_cache(maxsize=32)
def _activity_shape(path, size, modified, changed):
    import numpy as np
    with np.load(path) as data:
        return data['a'].shape


def _source_digest(names):
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode()); digest.update((root / name).read_bytes())
    return digest.hexdigest()


def solver_revision(method):
    names = ['metrics.py', 'algorithms/' + {'anc': 'cgap_sbl', 'cosamp': 'cosamp', 'omp': 'omp', 'sbl': 'sbl'}[method] + '.py']
    if method in {'sbl', 'anc'}:
        names.append('algorithms/posterior.py')
    return _source_digest(names)


def legacy_definition_matches(saved, current, cfg, method, condition=None):
    """Project recorded flat metadata onto the same method, without modifying it."""
    old, new = deepcopy(saved), deepcopy(current)
    for definition in (old, new):
        algorithms = definition.get('algorithms', {})
        definition['algorithms'] = solver_settings({'algorithms': algorithms}, method)
        if 'tuning' in definition:
            definition['tuning']['algorithms'] = solver_settings({'algorithms': definition['tuning'].get('algorithms', {})}, method)
        if 'selection' in definition:
            # The recorded selection retains its old definition; compare scientifically relevant tuning fields.
            selection = definition['selection']
            selection['definition']['algorithms'] = solver_settings({'algorithms': selection['definition'].get('algorithms', {})}, method)
    if condition is not None:
        for definition in (old, new):
            if list(condition) not in definition.get('communication', {}).get('conditions', []):
                return False
            definition.pop('conditions', None)
            definition['communication']['conditions'] = [list(condition)]
    return old == new
