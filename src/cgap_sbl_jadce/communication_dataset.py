from pathlib import Path
import hashlib
import json
from functools import lru_cache

import numpy as np

from cgap_sbl_jadce.paths import result_path, task_dir, condition_name
from cgap_sbl_jadce.file_cache import file_version, snapshot_cached
from cgap_sbl_jadce.io import save_npz
from cgap_sbl_jadce.models.realizations import Realization
from cgap_sbl_jadce.settings import reference_settings, communication_pairs, evaluation_settings


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
    from cgap_sbl_jadce.compatibility import _source_digest
    return {'schema': 3, 'code': _source_digest(['models/realizations.py', 'models/communication.py']),
            'M': int(cfg['system']['M']), 'N': int(cfg['system']['N']),
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
    if condition is not None:
        path = folder.parent / 'additional' / condition_name(condition) / f'{q:04d}' / 'inputs.npz'
        if path.exists():
            pair, realization = _load_condition_sample(path, channels)
            if pair != tuple(condition):
                raise ValueError(f'Communication archive has a different condition: {path}')
            return {pair: realization}
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
        pair, realization = _load_condition_sample(path, channels)
        result[pair] = realization
    return result


def _load_condition_sample(path, channels):
    with np.load(path) as archive:
        L, snr = archive['condition']
        pair = (int(L), float(snr))
        r = Realization(a=channels['a'], H=channels['H'], Z=channels['Z'],
                        Theta=archive['Theta'], Y=archive['Y'], W=archive['W'], sigma2=float(archive['sigma2']))
    return pair, r


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


def communication_checkpoint_definition(cfg, split, pair):
    base = communication_definition(cfg, split)
    base.pop('conditions')
    return {'definition': {'method': f'Communication inputs ({split})', **base},
            'condition': list(pair), 'samples': cfg['run'][f'num_{split}_samples']}


def add_communication_conditions(cfg, conditions, methods=(), show_progress=True, rerun=False, full_scope=False):
    from tqdm.auto import tqdm
    from cgap_sbl_jadce.additional_conditions import input_splits
    from cgap_sbl_jadce.checkpoints import Checkpoint, completed_outputs, output_hashes, prune_owned_checkpoints
    from cgap_sbl_jadce.compatibility import validate_preparation
    from cgap_sbl_jadce.dataset import load_activity_samples
    from cgap_sbl_jadce.models.realizations import communication_realization
    from cgap_sbl_jadce.io import save_json
    from cgap_sbl_jadce.provenance import save_task_snapshot
    from cgap_sbl_jadce.settings import communication_conditions
    from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
    pairs = communication_conditions(cfg, conditions)
    if any(method not in ESTIMATORS for method in methods):
        raise ValueError('Choose a registered algorithm.')
    splits = input_splits(cfg, methods)
    root = task_dir(cfg['run']['output_dir'], 'communication')
    summary_path = root / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    definitions = {}
    incompatible = []
    for split in splits:
        validate_preparation(cfg, 'activity', split)
        definitions[split] = communication_definition(cfg, split)
        if split in summary:
            try:
                validate_communication_dataset(cfg, split, [])
            except (OSError, ValueError, KeyError):
                incompatible.append(split)
    if incompatible and not (rerun and full_scope):
        raise ValueError('Shared communication inputs changed. Explicitly Rerun all conditions first.')
    if incompatible:
        from uuid import uuid4
        import shutil
        retired = root / '.retired' / uuid4().hex
        retired.mkdir(parents=True)
        save_json(retired / 'summary.json', summary)
        for split in incompatible:
            folder = root / split
            if folder.exists():
                shutil.move(str(folder), retired / split)
            summary.pop(split)
    if not rerun and all(set(pairs) <= {tuple(pair) for pair in summary.get(split, {}).get('definition', {}).get('conditions', [])}
                         for split in splits):
        for pair in pairs:
            for split in splits:
                communication_condition_identity(cfg, split, pair)
                prune_owned_checkpoints(root / split / 'additional' / condition_name(pair) / '.checkpoints',
                                        communication_checkpoint_definition(cfg, split, pair))
        return summary
    for split in splits:
        if split not in summary:
            summary[split] = {'samples': cfg['run'][f'num_{split}_samples'],
                              'definition': {**definitions[split], 'conditions': []},
                              'preview_condition': list(pairs[0])}
        if rerun:
            summary[split]['definition']['conditions'] = [value for value in summary[split]['definition']['conditions']
                                                         if tuple(value) not in pairs]
    summary['status'] = 'partial'
    save_json(summary_path, summary)
    save_task_snapshot(cfg, 'communication', status='started')
    activities = {split: load_activity_samples(result_path(cfg['run']['output_dir'], 'activity', f'{split}_activity.npz'))
                  for split in splits}
    for pair in pairs:
        L, snr = pair
        for split in splits:
            folder = root / split
            owned = folder / 'additional' / condition_name(pair)
            definition = communication_checkpoint_definition(cfg, split, pair)
            saved_pairs = {tuple(value) for value in summary[split]['definition']['conditions']}
            if pair in saved_pairs:
                communication_condition_identity(cfg, split, pair)
                prune_owned_checkpoints(owned / '.checkpoints', definition)
                continue
            checkpoint = Checkpoint(owned, definition)
            base = definitions[split]
            for q, activity in enumerate(tqdm(activities[split], desc=f'{split.capitalize()} inputs L={L} SNR={snr:g} dB',
                                            unit='realization', disable=not show_progress)):
                source = folder / f'{q:04d}'
                path = owned / f'{q:04d}' / 'inputs.npz'
                files = ([path, source / 'channels.npz'] if path.exists() else
                         [source / name for name in ('channels.npz', 'pilots.npz', 'observations.npz')])
                if completed_outputs(checkpoint, f'{q:04d}', files) is not None:
                    continue
                if len(activity.a) != base['N']:
                    raise ValueError('Saved activity dimensions differ from configuration; rerun activity generation.')
                r = communication_realization(activity, base['seed'], q, base['M'], L, snr)
                if not (source / 'channels.npz').exists():
                    save_communication_sample(folder, q, {pair: r})
                else:
                    try:
                        with np.load(source / 'channels.npz') as archive:
                            if any(not np.array_equal(archive[key], getattr(r, key)) for key in ('H', 'a', 'Z')):
                                raise ValueError('Saved shared channels/activity differ from deterministic inputs.')
                    except (OSError, ValueError, KeyError, EOFError):
                        if not (rerun and full_scope):
                            raise ValueError('Saved shared inputs are damaged; explicitly rerun all communication conditions.') from None
                        save_communication_sample(folder, q, {pair: r})
                    try:
                        saved = load_communication_sample(folder, q, pair).get(pair)
                    except (OSError, ValueError, KeyError, EOFError):
                        if path.exists():
                            saved = None
                        elif rerun and full_scope:
                            save_communication_sample(folder, q, {pair: r})
                            saved = r
                        else:
                            raise ValueError('Saved shared inputs are damaged; explicitly rerun all communication conditions.') from None
                    if saved is None or any(not np.array_equal(getattr(saved, key), getattr(r, key))
                                             for key in ('Theta', 'Y', 'W', 'sigma2')):
                        save_npz(path, condition=np.asarray(pair), Theta=r.Theta, Y=r.Y, W=r.W, sigma2=np.asarray(r.sigma2))
                files = ([path, source / 'channels.npz'] if path.exists() else
                         [source / name for name in ('channels.npz', 'pilots.npz', 'observations.npz')])
                checkpoint.save(f'{q:04d}', {'outputs': output_hashes(files)})
            summary[split]['definition']['conditions'] = [list(value) for value in sorted(saved_pairs | {pair})]
            reference = ((reference_settings(cfg)['reference_pilot_length'], reference_settings(cfg)['reference_snr_db'])
                         if split == 'tuning' else (cfg['evaluation']['pilot_length'], evaluation_settings(cfg)['snr_db']))
            summary[split]['preview_condition'] = list(reference if reference in saved_pairs | {pair} else pair)
            complete = all(set(communication_pairs(cfg, name)) <= {tuple(value) for value in summary[name]['definition']['conditions']}
                           for name in splits)
            summary['status'] = 'complete' if complete else 'partial'
            save_json(summary_path, summary)
    save_task_snapshot(cfg, 'communication', status=summary['status'], communication_datasets=summary)
    return summary
