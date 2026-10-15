from dataclasses import replace
from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cgap_sbl_jadce.checkpoints import Checkpoint
from cgap_sbl_jadce.io import atomic_path, save_yaml
from cgap_sbl_jadce.models.activity import generate_activity_dataset, calibrate_kappa_grid
from cgap_sbl_jadce.experiments import pipeline, estimators
from cgap_sbl_jadce.paths import result_path, task_dir


class Crash(RuntimeError):
    pass


def fail_after_commits(monkeypatch, limit):
    original = Checkpoint.save
    calls = {'count': 0}
    def save(checkpoint, *args, **kwargs):
        original(checkpoint, *args, **kwargs)
        calls['count'] += 1
        if calls['count'] == limit:
            raise Crash('simulated process failure after durable commit')
    monkeypatch.setattr(Checkpoint, 'save', save)
    return original


def test_failed_write_preserves_complete_file(tmp_path):
    path = tmp_path / 'summary.json'
    path.write_text('{"complete": true}')
    with pytest.raises(Crash):
        with atomic_path(path) as temporary:
            temporary.write_text('{"partial":')
            raise Crash()
    assert json.loads(path.read_text()) == {'complete': True}
    assert not list(tmp_path.glob('.tmp-*'))


def test_checkpoint_survives_fresh_process_and_rejects_changed_definition(tmp_path):
    cp = Checkpoint(tmp_path, {'seed': 1})
    cp.save('unit', {'done': 1}, gamma=np.arange(3))
    code = "from cgap_sbl_jadce.checkpoints import Checkpoint; import sys; c=Checkpoint(sys.argv[1],{'seed':1}); m,a=c.load('unit'); assert m['done']==1; assert a['gamma'].tolist()==[0,1,2]"
    subprocess.run([sys.executable, '-c', code, str(tmp_path)], check=True, capture_output=True, text=True)
    assert Checkpoint(tmp_path, {'seed': 2}).load('unit') is None
    (cp.folder / 'unit.npz').write_bytes(b'partial archive')
    assert cp.load('unit') is None


def test_activity_restart_retains_seed_sequence(tmp_path, monkeypatch):
    args = dict(seed=123, num_samples=5, N=20, V=2, R=40, kappa=3.78, D=15,
                S=3, exact_sparsity=False, show_progress=False)
    expected = generate_activity_dataset(**args)
    cp = Checkpoint(tmp_path, args)
    original = fail_after_commits(monkeypatch, 2)
    with pytest.raises(Crash):
        generate_activity_dataset(**args, checkpoint=cp)
    monkeypatch.setattr(Checkpoint, 'save', original)
    from cgap_sbl_jadce.models import activity
    generate = activity.generate_activity_sample
    calls = []
    def count(*args, **kwargs):
        calls.append(1)
        return generate(*args, **kwargs)
    monkeypatch.setattr(activity, 'generate_activity_sample', count)
    resumed = generate_activity_dataset(**args, checkpoint=Checkpoint(tmp_path, args))
    assert len(calls) == 3
    for a, b in zip(resumed, expected):
        for field in ['a', 'device_locations', 'event_locations', 'activation_probabilities']:
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_calibration_restart_restores_rng_and_running_sums(tmp_path, monkeypatch):
    args = dict(seed=77, num_samples=25, N=20, V=2, R=40, D=15,
                kappa_values=[1., 3.], batch_size=5, show_progress=False)
    expected = calibrate_kappa_grid(**args)
    cp = Checkpoint(tmp_path, args)
    original = fail_after_commits(monkeypatch, 2)
    with pytest.raises(Crash):
        calibrate_kappa_grid(**args, checkpoint=cp)
    monkeypatch.setattr(Checkpoint, 'save', original)
    from cgap_sbl_jadce.models import activity
    disk = activity._uniform_disk_batch
    calls = []
    def count(*args):
        calls.append(1)
        return disk(*args)
    monkeypatch.setattr(activity, '_uniform_disk_batch', count)
    resumed = calibrate_kappa_grid(**args, checkpoint=Checkpoint(tmp_path, args))
    assert resumed == expected
    assert len(calls) == 6


@pytest.mark.parametrize('stage,task,operation', [
    ('communication_stage', 'communication', 'communication_realization'),
    ('spatial_dependence_stage', 'spatial_dependence', 'build_C'),
])
def test_generation_restart_skips_committed_units_and_fresh_rerun_rebuilds_outputs(saved_data, monkeypatch, stage, task, operation):
    original_save = fail_after_commits(monkeypatch, 1)
    with pytest.raises(Crash):
        getattr(pipeline, stage)(saved_data, show_progress=False)
    monkeypatch.setattr(Checkpoint, 'save', original_save)
    original = getattr(pipeline, operation)
    calls = []
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, operation, count)
    getattr(pipeline, stage)(saved_data, show_progress=False, resume=True)

    assert len(calls) == (7 if task == 'communication' else 3)
    assert not list(task_dir(saved_data['run']['output_dir'], task).rglob('.checkpoints/*/*.npz'))
    calls.clear()
    folder = task_dir(saved_data['run']['output_dir'], task) / 'tuning'
    path = folder / '0000' / 'channels.npz' if task == 'communication' else folder / '0000.npz'
    path.write_bytes(b'interrupted or replaced archive')
    getattr(pipeline, stage)(saved_data, show_progress=False)
    assert len(calls) == (8 if task == 'communication' else 4)
    assert path.stat().st_size > 100


def test_tuning_restart_reuses_completed_solver_outputs(saved_data, monkeypatch):
    method = 'anc'
    spec = estimators.ESTIMATORS[method]
    calls = []
    def solve(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise Crash()
        return spec.solve(*args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, method, replace(spec, solve=solve))
    with pytest.raises(Crash):
        estimators.tune_estimator_stage(saved_data, method, show_progress=False)
    assert len(calls) == 3
    calls.clear()
    def resumed(*args, **kwargs):
        calls.append(1)
        return spec.solve(*args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, method, replace(spec, solve=resumed))
    selected = estimators.tune_estimator_stage(saved_data, method, show_progress=False, resume=True)
    assert len(calls) == 6
    scores = pd.read_csv(result_path(saved_data['run']['output_dir'], method, 'tuning/alpha_beta.csv'))
    assert len(scores) == 4
    assert selected['parameters']['tau'] in np.arange(201) / 1000


def test_evaluation_continues_in_fresh_process_with_identical_scores(saved_data, monkeypatch, tmp_path):
    spec = estimators.ESTIMATORS['omp']
    calls = []
    def solve(*args, **kwargs):
        calls.append(1)
        if len(calls) == 4:
            raise Crash()
        return spec.solve(*args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=solve))
    with pytest.raises(Crash):
        estimators.run_estimator_stage(saved_data, 'omp', show_progress=False)
    cp_folder = task_dir(saved_data['run']['output_dir'], 'omp') / 'run'
    checkpoint = next(cp_folder.glob('*/.checkpoints/*/0000.npz'))
    with np.load(checkpoint) as data:
        committed = json.loads(str(data['_metadata']))['row']
    assert committed['q'] == 0
    input_path = tmp_path / 'resume.yaml'
    save_yaml(input_path, saved_data)
    code = "from cgap_sbl_jadce.config import load_config; from cgap_sbl_jadce.experiments.estimators import run_estimator_stage; import sys; run_estimator_stage(load_config(sys.argv[1]),'omp',show_progress=False,resume=True)"
    subprocess.run([sys.executable, '-c', code, str(input_path)], check=True, capture_output=True, text=True)
    resumed = pd.read_csv(result_path(saved_data['run']['output_dir'], 'omp', 'per_sample.csv'))
    unit = resumed[(resumed.q == committed['q']) & (resumed.L == committed['L']) & (resumed.snr_db == committed['snr_db'])]
    np.testing.assert_allclose(unit.runtime_s, committed['runtime_s'])
    assert len(resumed) == 8
    assert not resumed.duplicated(['q', 'L', 'snr_db', 'sweep']).any()
    cfg = {**saved_data, 'run': {**saved_data['run'], 'output_dir': str(tmp_path / 'uninterrupted')}}
    for task in ['activity', 'communication', 'spatial_dependence']:
        shutil.copytree(task_dir(saved_data['run']['output_dir'], task), task_dir(cfg['run']['output_dir'], task))
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', spec)
    estimators.run_estimator_stage(cfg, 'omp', show_progress=False)
    expected = pd.read_csv(result_path(cfg['run']['output_dir'], 'omp', 'per_sample.csv'))
    pd.testing.assert_frame_equal(resumed.drop(columns='runtime_s'), expected.drop(columns='runtime_s'))


def test_convergence_reuses_completed_trace_after_failure(saved_data, monkeypatch):
    for method in ['sbl', 'anc']:
        estimators.tune_estimator_stage(saved_data, method, show_progress=False)
        estimators.run_estimator_stage(saved_data, method, show_progress=False)
    spec = estimators.ESTIMATORS['anc']
    monkeypatch.setitem(estimators.ESTIMATORS, 'anc', replace(spec, solve=lambda *a, **k: (_ for _ in ()).throw(Crash())))
    with pytest.raises(Crash):
        estimators.convergence_stage(saved_data, show_progress=False)
    monkeypatch.setitem(estimators.ESTIMATORS, 'anc', spec)
    sb = estimators.ESTIMATORS['sbl']
    monkeypatch.setitem(estimators.ESTIMATORS, 'sbl', replace(sb, solve=lambda *a, **k: pytest.fail('Repeated completed trace')))
    summary = estimators.convergence_stage(saved_data, show_progress=False, resume=True)
    assert set(summary['scores']) == {'sbl', 'anc'}


def test_abrupt_exit_during_write_keeps_previous_checkpoint(tmp_path):
    cp = Checkpoint(tmp_path, {'task': 'example'})
    cp.save('unit', {'completed': 1}, gamma=np.arange(3))
    path = cp.folder / 'unit.npz'
    code = "from cgap_sbl_jadce.io import atomic_path; import os,sys; p=atomic_path(sys.argv[1]); temporary=p.__enter__(); temporary.write_bytes(b'incomplete'); os._exit(17)"
    result = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True)
    assert result.returncode == 17
    metadata, arrays = Checkpoint(tmp_path, {'task': 'example'}).load('unit')
    assert metadata == {'completed': 1}
    np.testing.assert_array_equal(arrays['gamma'], np.arange(3))


def test_changed_archive_bytes_cannot_reuse_evaluation_scores(saved_data, monkeypatch):
    from cgap_sbl_jadce.io import save_npz
    estimators.run_estimator_stage(saved_data, 'omp', show_progress=False)
    path = task_dir(saved_data['run']['output_dir'], 'communication') / 'evaluation' / '0000' / 'observations.npz'
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key in arrays:
        if key.endswith('_Y'):
            arrays[key] = arrays[key] + .1
    save_npz(path, **arrays)
    spec = estimators.ESTIMATORS['omp']
    calls = []
    def solve(*args, **kwargs):
        calls.append(1)
        return spec.solve(*args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=solve))
    estimators.run_estimator_stage(saved_data, 'omp', show_progress=False)
    assert len(calls) == 6
