"""Saved sparse spatial dependence matrices."""
from pathlib import Path
import hashlib
import json
from scipy.sparse import load_npz
from cgap_sbl_jadce.config import activity_spatial_dependence_parameters
from cgap_sbl_jadce.paths import result_path, task_dir


def spatial_dependence_definition(cfg, split):
    path = result_path(cfg['run']['output_dir'], 'activity', f'{split}_activity.npz')
    if not path.exists():
        raise FileNotFoundError(f'Run activity generation first: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'schema': 1, 'N': int(cfg['system']['N']),
            **activity_spatial_dependence_parameters(cfg), 'activity_sha256': digest}


def validate_spatial_dependence_dataset(cfg, split):
    path = result_path(cfg['run']['output_dir'], 'spatial_dependence', 'summary.json')
    if not path.exists():
        raise FileNotFoundError('Run spatial dependence to save spatial dependence matrices first.')
    summary = json.loads(path.read_text())
    if summary.get('status') != 'complete' or summary.get(split, {}).get('definition') != spatial_dependence_definition(cfg, split):
        raise ValueError(f'Saved {split} spatial matrices are missing or incompatible; rerun spatial dependence.')
    return task_dir(cfg['run']['output_dir'], 'spatial_dependence') / split


def load_spatial_dependence_matrix(folder, q):
    path = Path(folder) / f'{q:04d}.npz'
    if not path.exists():
        raise FileNotFoundError(f'Spatial matrix missing; rerun spatial dependence: {path}')
    return load_npz(path).tocsr()


def load_tuning_spatial_dependence(cfg, count):
    folder = validate_spatial_dependence_dataset(cfg, 'tuning')
    return [load_spatial_dependence_matrix(folder, q) for q in range(count)]
