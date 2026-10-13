"""Saved communication archives, shared by all downstream experiments."""
from pathlib import Path
import hashlib
import json

import numpy as np

from cgap_sbl_jadce.paths import result_path, task_dir
from cgap_sbl_jadce.io import save_npz
from cgap_sbl_jadce.models.realizations import Realization


def communication_definition(cfg, split):
    names = [split]
    hashes = {}
    for name in names:
        path = result_path(cfg['run']['output_dir'], 'activity', f'{name}_activity.npz')
        if not path.exists():
            raise FileNotFoundError(f'Run activity generation first: {path}')
        with path.open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    tune = cfg['tuning']
    conditions = cfg.get('communication', {}).get('tuning_conditions',
                    [(int(tune['reference_pilot_length']), float(tune['reference_snr_db']))])
    if split == 'evaluation':
        from cgap_sbl_jadce.experiments.estimators import evaluation_conditions
        conditions = sorted({(L, snr) for L, snr, _ in evaluation_conditions(cfg)} |
                            {(int(cfg['evaluation']['pilot_length']), float(tune['reference_snr_db']))})
    return {'schema': 2, 'M': int(cfg['system']['M']), 'N': int(cfg['system']['N']),
            'seed': cfg['seeds'][f'communication_{split}'], 'activity_sha256': hashes,
            'conditions': [list(condition) for condition in conditions]}


def validate_communication_dataset(cfg, split):
    path = result_path(cfg['run']['output_dir'], 'communication', 'summary.json')
    if not path.exists():
        raise FileNotFoundError('Run communication generation to generate the full communication dataset first.')
    summary = json.loads(path.read_text())
    if summary.get(split, {}).get('definition') != communication_definition(cfg, split):
        raise ValueError(f'Saved {split} communication data are missing or differ from the setup/activity. Rerun communication generation.')
    return task_dir(cfg['run']['output_dir'], 'communication') / split


def save_communication_sample(folder, q, realizations):
    """Save H/Z/a once, pilots once per L, and Y/W/sigma2 per condition."""
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


def load_communication_sample(folder, q):
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
            condition = key.removesuffix('_Y')
            L, snr = condition[1:].split('_snr')
            result[(int(L), float(snr))] = Realization(
                a=channels['a'], H=channels['H'], Z=channels['Z'],
                Theta=pilots[f'L{L}'], Y=archive[key], W=archive[f'{condition}_W'],
                sigma2=float(archive[f'{condition}_sigma2']))
    return result


def load_tuning_communication(cfg, count):
    folder = validate_communication_dataset(cfg, 'tuning')
    L = int(cfg['tuning']['reference_pilot_length'])
    snr = float(cfg['tuning']['reference_snr_db'])
    return [load_communication_sample(folder, q)[(L, snr)] for q in range(count)]
