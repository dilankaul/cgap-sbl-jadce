import hashlib
import os

import numpy as np

from cgap_sbl_jadce.communication_dataset import communication_condition_identity
from cgap_sbl_jadce.file_cache import validation_snapshot
from cgap_sbl_jadce.paths import run_path, task_dir


def test_cached_identity_matches_saved_arrays_and_tracks_condition_changes(saved_data):
    pair, other = (7, 8.), (7, 12.)
    original = communication_condition_identity(saved_data, 'evaluation', pair)
    unrelated = communication_condition_identity(saved_data, 'evaluation', other)
    folder = task_dir(saved_data['run']['output_dir'], 'communication') / 'evaluation'
    digest = hashlib.sha256()
    for q in range(2):
        for path, keys in [(folder / f'{q:04d}/channels.npz', ('H', 'a', 'Z')),
                           (folder / f'additional/L0007_snr8dB/{q:04d}/inputs.npz', ('Theta', 'Y', 'W', 'sigma2'))]:
            group = hashlib.sha256()
            with np.load(path) as archive:
                for key in keys:
                    value = np.ascontiguousarray(archive[key])
                    group.update(key.encode())
                    group.update(str(value.dtype).encode())
                    group.update(str(value.shape).encode())
                    group.update(value.tobytes())
            digest.update(group.hexdigest().encode())
    assert original['arrays_sha256'] == digest.hexdigest()
    path = folder / 'additional/L0007_snr8dB/0000/inputs.npz'
    contents, stat = path.read_bytes(), path.stat()
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['Y'][0, 0] += 1
    np.savez_compressed(path, **arrays)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert communication_condition_identity(saved_data, 'evaluation', pair) != original
    assert communication_condition_identity(saved_data, 'evaluation', other) == unrelated
    path.write_bytes(contents)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert communication_condition_identity(saved_data, 'evaluation', pair) == original


def test_refresh_shares_input_checks_and_next_refresh_rechecks_files(saved_data, monkeypatch):
    from pathlib import Path
    path = task_dir(saved_data['run']['output_dir'], 'communication') / 'evaluation/0000/channels.npz'
    stat = Path.stat
    calls = []
    def count(source, *args, **kwargs):
        if source == path:
            calls.append(source)
        return stat(source, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', count)
    for refresh in range(2):
        with validation_snapshot():
            first = communication_condition_identity(saved_data, 'evaluation', (7, 8.))
            communication_condition_identity(saved_data, 'evaluation', (7, 12.))
            assert communication_condition_identity(saved_data, 'evaluation', (7, 8.)) == first
        assert len(calls) == refresh + 1


def test_cached_results_detect_modified_output_and_restore_eligibility(saved_data):
    from cgap_sbl_jadce.checkpoints import _archive_digest
    from cgap_sbl_jadce.condition_results import completed_conditions
    from cgap_sbl_jadce.experiments.estimators import run_estimator_stage
    run_estimator_stage(saved_data, 'omp', False, conditions=[(7, 8.)])
    assert len(completed_conditions(saved_data, 'omp')) == 1
    hashes = _archive_digest.cache_info().misses
    assert len(completed_conditions(saved_data, 'omp')) == 1
    assert _archive_digest.cache_info().misses == hashes
    path = run_path(saved_data, 'omp', 'per_sample.csv', (7, 8.))
    contents, stat = path.read_bytes(), path.stat()
    path.write_bytes(contents.replace(b'MMV-OMP', b'MMV-BAD'))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert completed_conditions(saved_data, 'omp') == []
    path.write_bytes(contents)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert len(completed_conditions(saved_data, 'omp')) == 1
    saved_data['seeds']['communication_evaluation'] += 1
    assert completed_conditions(saved_data, 'omp') == []
    saved_data['seeds']['communication_evaluation'] -= 1
    assert len(completed_conditions(saved_data, 'omp')) == 1
