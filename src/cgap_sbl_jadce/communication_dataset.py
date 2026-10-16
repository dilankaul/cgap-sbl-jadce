from pathlib import Path
import hashlib
import json
from functools import lru_cache

import numpy as np

from cgap_sbl_jadce.paths import result_path, task_dir, condition_name
from cgap_sbl_jadce.file_cache import file_version, snapshot_cached
from cgap_sbl_jadce.io import save_npz
from cgap_sbl_jadce.models.realizations import Realization
from cgap_sbl_jadce.settings import reference_settings, communication_pairs


def communication_definition(cfg, split):
    names = [split]
    hashes = {}
    for name in names:
        path = result_path(cfg['run']['output_dir'], 'activity', f'{name}_activity.npz')
        if not path.exists():
            raise FileNotFoundError(f'Run activity generation first: {path}')
        from cgap_sbl_jadce.checkpoints import _archive_digest
        stat = path.stat()
        hashes[name] = _archive_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns).hex()
    conditions = communication_pairs(cfg, split)
    return {'schema': 2, 'M': int(cfg['system']['M']), 'N': int(cfg['system']['N']),
            'seed': cfg['seeds'][f'communication_{split}'], 'activity_sha256': hashes,
            'conditions': [list(condition) for condition in conditions]}


def validate_communication_dataset(cfg, split, conditions=None):
    path = result_path(cfg['run']['output_dir'], 'communication', 'summary.json')
    if not path.exists():
        raise FileNotFoundError('Run communication generation to generate the full communication dataset first.')
    summary = json.loads(path.read_text())
    from cgap_sbl_jadce.compatibility import require_match, validate_preparation, IncompatibleResults
    validate_preparation(cfg, 'activity', split)
    expected = communication_definition(cfg, split)
    saved = summary.get(split, {}).get('definition', {})
    require_match({key: saved.get(key) for key in expected if key != 'conditions'},
                  {key: value for key, value in expected.items() if key != 'conditions'}, f'{split.capitalize()} communication (Rerun communication generation)')
    required = {tuple(pair) for pair in (conditions if conditions is not None else expected['conditions'])}
    if not required <= {tuple(pair) for pair in saved.get('conditions', [])}:
        raise IncompatibleResults(f'{split.capitalize()} communication is missing requested L/SNR conditions; run communication generation.')
    if summary.get(split, {}).get('samples') != cfg['run'][f'num_{split}_samples']:
        raise IncompatibleResults(f'{split.capitalize()} communication incompatible: changed samples.')
    return task_dir(cfg['run']['output_dir'], 'communication') / split


def save_communication_sample(folder, q, realizations):
    folder = Path(folder) / f'{q:04d}'
    first = next(iter(realizations.values()))
    pilots = {}
    observations = {}
    for (L, snr), r in realizations.items():
        pilots[f'L{L}'] = r.Theta
        key = f'L{L}_snr{snr:g}'
        observations.update({f'{key}_Y': r.Y, f'{key}_W': r.W,
                             f'{key}_sigma2': np.asarray(r.sigma2)})
    save_npz(folder / 'channels.npz', H=first.H, a=first.a, Z=first.Z)
    save_npz(folder / 'pilots.npz', **pilots)
    save_npz(folder / 'observations.npz', **observations)


def load_communication_sample(folder, q, condition=None):
    folder = Path(folder) / f'{q:04d}'
    if not (folder / 'channels.npz').exists():
        raise FileNotFoundError(f'Communication realization missing; rerun communication generation: {folder}')
    with np.load(folder / 'channels.npz') as archive:
        channels = {key: archive[key] for key in archive.files}
    with np.load(folder / 'pilots.npz') as archive:
        pilots = {key: archive[key] for key in archive.files}
    result = {}
    with np.load(folder / 'observations.npz') as archive:
        for key in archive.files:
            if not key.endswith('_Y'):
                continue
            name = key.removesuffix('_Y')
            L, snr = name[1:].split('_snr')
            if condition is not None and (int(L), float(snr)) != tuple(condition):
                continue
            result[(int(L), float(snr))] = Realization(
                a=channels['a'], H=channels['H'], Z=channels['Z'],
                Theta=pilots[f'L{L}'], Y=archive[key], W=archive[f'{name}_W'],
                sigma2=float(archive[f'{name}_sigma2']))
    extra = folder.parent / 'additional'
    sources = [extra / condition_name(condition)] if condition is not None else extra.glob('L*_snr*dB')
    for source in sources:
        path = source / f'{q:04d}' / 'inputs.npz'
        if not path.exists():
            continue
        with np.load(path) as archive:
            pair = tuple(archive['condition'])
            pair = (int(pair[0]), float(pair[1]))
            result[pair] = Realization(a=channels['a'], H=channels['H'], Z=channels['Z'],
                Theta=archive['Theta'], Y=archive['Y'], W=archive['W'], sigma2=float(archive['sigma2']))
    return result


def load_tuning_communication(cfg, count):
    folder = validate_communication_dataset(cfg, 'tuning', [(reference_settings(cfg)['reference_pilot_length'], reference_settings(cfg)['reference_snr_db'])])
    L = int(reference_settings(cfg)['reference_pilot_length'])
    snr = float(reference_settings(cfg)['reference_snr_db'])
    return [load_communication_sample(folder, q, (L, snr))[(L, snr)] for q in range(count)]


@lru_cache(maxsize=32768)
def _array_digest(path, size, modified, changed, keys):
    digest = hashlib.sha256()
    with np.load(path) as archive:
        for key in keys:
            if key not in archive.files:
                raise ValueError(f'Missing communication input {key} in {path}')
            value = np.ascontiguousarray(archive[key])
            digest.update(key.encode()); digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _condition_digest(sources):
    digest = hashlib.sha256()
    for version, keys in sources:
        digest.update(_array_digest(*version, keys).encode())
    return digest.hexdigest()


@snapshot_cached
def _base_versions(folder, count):
    return tuple(tuple(file_version(f'{folder}/{q:04d}/{name}.npz')
                       for name in ('channels', 'pilots', 'observations')) for q in range(count))


@snapshot_cached
def communication_condition_identity(cfg, split, pair):
    folder = validate_communication_dataset(cfg, split, [pair]).resolve()
    L, snr = pair
    key = f'L{L}_snr{snr:g}'
    additional = folder / 'additional' / condition_name(pair)
    has_additional = additional.is_dir()
    sources = []
    count = cfg['run'][f'num_{split}_samples']
    if not has_additional:
        for channels, pilots, observations in _base_versions(str(folder), count):
            sources.extend([(channels, ('H', 'a', 'Z')), (pilots, (f'L{L}',)),
                            (observations, tuple(f'{key}_{suffix}' for suffix in ('Y', 'W', 'sigma2')))])
    for q in range(count) if has_additional else ():
        sample = folder / f'{q:04d}'
        sources.append((file_version(sample / 'channels.npz'), ('H', 'a', 'Z')))
        extra = additional / f'{q:04d}' / 'inputs.npz' if has_additional else None
        if extra is not None and extra.exists():
            sources.append((file_version(extra), ('Theta', 'Y', 'W', 'sigma2')))
        else:
            sources.extend([(file_version(sample / 'pilots.npz'), (f'L{L}',)),
                            (file_version(sample / 'observations.npz'), tuple(f'{key}_{suffix}' for suffix in ('Y', 'W', 'sigma2')))])
    definition = communication_definition(cfg, split)
    definition.pop('conditions')
    return {**definition, 'condition': list(pair), 'arrays_sha256': _condition_digest(tuple(sources))}


def add_communication_conditions(cfg, conditions, methods=(), show_progress=True):
    from tqdm.auto import tqdm
    from cgap_sbl_jadce.additional_conditions import additional_conditions, validate_pair, input_splits
    from cgap_sbl_jadce.checkpoints import Checkpoint, completed_outputs, output_hashes
    from cgap_sbl_jadce.dataset import load_activity_samples
    from cgap_sbl_jadce.models.realizations import communication_realization
    from cgap_sbl_jadce.io import save_json
    from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
    pairs = list(dict.fromkeys(validate_pair(pair) for pair in conditions))
    optional = set(additional_conditions(cfg))
    allowed = optional if methods else optional | set(communication_pairs(cfg, 'tuning')) | set(communication_pairs(cfg, 'evaluation'))
    if not pairs or not set(pairs) <= allowed or any(method not in ESTIMATORS for method in methods):
        raise ValueError('Choose setup conditions or add optional conditions on an algorithm Run page first.')
    splits = input_splits(cfg, methods)
    folders = {split: validate_communication_dataset(cfg, split, []) for split in splits}
    summary_path = result_path(cfg['run']['output_dir'], 'communication', 'summary.json')
    summary = json.loads(summary_path.read_text())
    for split, folder in folders.items():
        base = communication_definition(cfg, split)
        base.pop('conditions')
        activities = load_activity_samples(result_path(cfg['run']['output_dir'], 'activity', f'{split}_activity.npz'))
        for pair in pairs:
            if not methods and pair not in optional | set(communication_pairs(cfg, split)):
                continue
            if pair in {tuple(value) for value in summary[split]['definition']['conditions']}:
                try:
                    communication_condition_identity(cfg, split, pair)
                except (OSError, ValueError, KeyError):
                    pass
                else:
                    continue
            L, snr = pair
            checkpoint = Checkpoint(folder, {'definition': {'method': f'Communication inputs ({split})', **base},
                                              'condition': list(pair), 'samples': len(activities)})
            for q, activity in enumerate(tqdm(activities, desc=f'Additional {split} inputs L={L} SNR={snr:g} dB',
                                            unit='realization', disable=not show_progress)):
                source = folder / f'{q:04d}'
                path = folder / 'additional' / condition_name(pair) / f'{q:04d}' / 'inputs.npz'
                if completed_outputs(checkpoint, f'{q:04d}', [path]) is not None:
                    continue
                r = communication_realization(activity, base['seed'], q, base['M'], L, snr)
                with np.load(source / 'channels.npz') as archive:
                    if any(not np.array_equal(archive[key], getattr(r, key)) for key in ('H', 'a', 'Z')):
                        raise ValueError('Saved shared channels/activity differ from deterministic inputs; rerun communication generation.')
                with np.load(source / 'pilots.npz') as archive:
                    if f'L{L}' in archive and not np.array_equal(archive[f'L{L}'], r.Theta):
                        raise ValueError('Saved shared pilots differ from deterministic inputs; rerun communication generation.')

                values = {'condition': np.asarray(pair), 'Theta': r.Theta, 'Y': r.Y, 'W': r.W, 'sigma2': np.asarray(r.sigma2)}
                unchanged = False
                if path.exists():
                    with np.load(path) as archive:
                        unchanged = all(key in archive and np.array_equal(archive[key], value) for key, value in values.items())
                if not unchanged:
                    save_npz(path, **values)
                checkpoint.save(f'{q:04d}', {'outputs': output_hashes([path])})
            saved_pairs = {tuple(value) for value in summary[split]['definition']['conditions']}
            summary[split]['definition']['conditions'] = [list(value) for value in sorted(saved_pairs | {pair})]
            save_json(summary_path, summary)
            checkpoint.clear(prune=True)
    return summary
