from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import pytest

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.experiments import estimators
from cgap_sbl_jadce.gui.jobs import Jobs
from cgap_sbl_jadce.io import save_yaml
from cgap_sbl_jadce.paths import run_path, result_path, task_dir
from cgap_sbl_jadce.recovery import begin_recovery, recovery_path


class Failure(RuntimeError):
    pass


def test_conditions_complete_individually_and_rerun_does_not_use_finished_checkpoints(saved_data, monkeypatch):
    original = estimators.ESTIMATORS['omp']
    calls = []
    def solve(r, *args, **kwargs):
        calls.append((r.Theta.shape[0], r.sigma2))
        return original.solve(r, *args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(original, solve=solve))
    for pair in [(7, 8), (7, 12), (5, 12)]:
        summary = estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[pair])
        folder = run_path(saved_data, 'omp', condition=pair)
        assert json.loads((folder / 'summary.json').read_text())['status'] == 'complete'
        assert (folder / 'figures/realization_scores.png').exists()
        assert not list(folder.rglob('*.npz'))
    assert summary['status'] == 'complete' and len(summary['completed_pairs']) == 3
    assert len(calls) == 6
    untouched = run_path(saved_data, 'omp', 'per_sample.csv', (5, 12)).read_bytes()
    estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[(7, 8)])
    assert len(calls) == 8
    assert run_path(saved_data, 'omp', 'per_sample.csv', (5, 12)).read_bytes() == untouched
    estimators.run_estimator_stage(saved_data, 'omp', False)
    assert len(calls) == 14


def test_failure_keeps_only_unfinished_condition_and_resume_preserves_runtime(saved_data, monkeypatch):
    original = estimators.ESTIMATORS['omp']
    calls = []
    def solve(*args, **kwargs):
        calls.append(1)
        if len(calls) == 4:
            raise Failure()
        return original.solve(*args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(original, solve=solve))
    with pytest.raises(Failure):
        estimators.run_estimator_stage(saved_data, 'omp', False)
    first = run_path(saved_data, 'omp', condition=(7, 8))
    pending = run_path(saved_data, 'omp', condition=(7, 12))
    assert json.loads((first / 'summary.json').read_text())['status'] == 'complete'
    assert not list(first.glob('.checkpoints/*/*.npz'))
    unit = next(pending.glob('.checkpoints/*/0000.npz'))
    with np.load(unit) as archive:
        row = json.loads(str(archive['_metadata']))['row']
    calls.clear()
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(original, solve=lambda *a, **k: (calls.append(1), original.solve(*a, **k))[1]))
    summary = estimators.run_estimator_stage(saved_data, 'omp', False, resume=True)
    assert len(calls) == 3 and summary['status'] == 'complete'
    data = pd.read_csv(pending / 'per_sample.csv')
    np.testing.assert_allclose(data[data.q.eq(0)].runtime_s, row['runtime_s'])
    assert not list(run_path(saved_data, 'omp').rglob('.checkpoints/*/*.npz'))


def test_failed_figure_retains_numerical_checkpoint_for_resume(saved_data, monkeypatch):
    from cgap_sbl_jadce.figures import estimator_condition
    render = estimator_condition.plot_condition_scores
    monkeypatch.setattr(estimator_condition, 'plot_condition_scores', lambda *a, **k: (_ for _ in ()).throw(Failure()))
    with pytest.raises(Failure):
        estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[[7, 8]])
    assert list(run_path(saved_data, 'omp').rglob('.checkpoints/*/*.npz'))
    monkeypatch.setattr(estimator_condition, 'plot_condition_scores', render)
    original = estimators.ESTIMATORS['omp']
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(original, solve=lambda *a, **k: pytest.fail('Repeated saved solver unit')))
    estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[[7, 8]], resume=True)
    assert not list(run_path(saved_data, 'omp').rglob('.checkpoints/*/*.npz'))


def test_gui_can_recover_after_copy_without_local_jobs(saved_data, tmp_path, monkeypatch):
    cfg = dict(saved_data, experiment={'name': 'Portable setup'})
    config = tmp_path / 'config.yaml'
    save_yaml(config, cfg)
    cfg = load_config(config)
    request = {'kind': 'task', 'config': str(config), 'task': 'omp', 'action': 'run', 'conditions': [[7, 8]]}
    begin_recovery(request)
    original = estimators.ESTIMATORS['omp']
    calls = []
    def solve(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise Failure()
        return original.solve(*args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(original, solve=solve))
    with pytest.raises(Failure):
        estimators.run_estimator_stage(cfg, 'omp', False, conditions=[[7, 8]])
    clone = tmp_path.parent / (tmp_path.name + '-copy')
    shutil.copytree(tmp_path, clone, ignore=shutil.ignore_patterns('.gui'))
    jobs = Jobs(clone)
    config = clone / 'config.yaml'
    record = jobs.task_job(config, 'omp')
    assert record['portable'] and record['status'] == 'interrupted'
    resumed = jobs.resume(record['id'])
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = jobs.records()[0]
            if state['status'] not in {'queued', 'running'}:
                break
            time.sleep(.05)
        assert state['status'] == 'completed', jobs.log(resumed['id'])[0]
        assert not recovery_path(config, 'omp').exists()
        assert not list((clone / 'omp').rglob('.checkpoints/*/*.npz'))
        assert run_path(load_config(config), 'omp', 'per_sample.csv', (7, 8)).exists()
    finally:
        if jobs.busy():
            jobs.control(resumed['id'], 'stop')
        jobs.children[resumed['id']].wait(timeout=10)


def test_condition_graphs_and_export_use_numerical_results(saved_data, tmp_path, monkeypatch):
    from cgap_sbl_jadce.gui.charts import interactive_figures
    from cgap_sbl_jadce.gui.previews import render_preview
    estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[[7, 8]])
    config = tmp_path / 'config.yaml'
    save_yaml(config, saved_data)
    request = {'config': str(config), 'category': 'omp_run', 'condition': [7, 8]}
    monkeypatch.setattr(estimators, 'run_estimator_stage', lambda *a, **k: pytest.fail('Graph ran solver'))
    charts = interactive_figures(request)
    assert len(charts) == 3 and charts[1][1].layout.yaxis.type == 'log'
    assert list(charts[0][1].data[0].x) == [0, 1]
    paths = render_preview(request, tmp_path / 'exports')
    assert {path.suffix for path in paths} == {'.pdf', '.png'}


def test_combined_tables_do_not_replace_missing_condition_results(saved_data):
    estimators.run_estimator_stage(saved_data, 'omp', False)
    path = result_path(saved_data['run']['output_dir'], 'omp', 'per_sample.csv')
    original = path.read_bytes()
    shutil.rmtree(run_path(saved_data, 'omp'))
    assert estimators.completed_methods(saved_data) == []
    assert not run_path(saved_data, 'omp').exists()
    assert path.read_bytes() == original


def test_crash_during_final_cleanup_resumes_without_recomputing(saved_data, tmp_path, monkeypatch):
    from cgap_sbl_jadce.checkpoints import Checkpoint
    from cgap_sbl_jadce.tasks import run_task
    from cgap_sbl_jadce.recovery import finish_recovery
    config = tmp_path / 'input.yaml'
    save_yaml(config, saved_data)
    request = {'kind': 'task', 'config': str(config), 'task': 'communication', 'action': 'run'}
    begin_recovery(request)
    cfg = dict(saved_data, task_runtime={'input_config': str(config)})
    clear = Checkpoint.clear
    def crash(checkpoint, **kwargs):
        if checkpoint.units:
            raise Failure('shutdown during cleanup')
        return clear(checkpoint, **kwargs)
    monkeypatch.setattr(Checkpoint, 'clear', crash)
    with pytest.raises(Failure):
        run_task('communication', cfg, show_progress=False)
    assert json.loads(recovery_path(config, 'communication').read_text())['phase'] == 'results_saved'
    monkeypatch.setattr(Checkpoint, 'clear', clear)
    from cgap_sbl_jadce.experiments import pipeline
    monkeypatch.setattr(pipeline, 'communication_realization', lambda *a, **k: pytest.fail('Repeated completed generator'))
    paths = run_task('communication', cfg, show_progress=False, resume=True)
    assert all(path.exists() for path in paths)
    assert not list(task_dir(saved_data['run']['output_dir'], 'communication').rglob('.checkpoints/*/*.npz'))
    finish_recovery(request)


def test_disjoint_conditions_can_run_while_same_algorithm_is_paused(saved_data, tmp_path):
    config = tmp_path / 'input.yaml'
    save_yaml(config, saved_data)
    jobs = Jobs(tmp_path)
    first = jobs.start({'kind': 'task', 'config': str(config), 'task': 'omp', 'action': 'run', 'conditions': [[7, 8]]})
    second = None
    def wait(job_id, statuses):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            record = next(record for record in jobs.records() if record['id'] == job_id)
            if record['status'] in statuses:
                return record
            time.sleep(.05)
        pytest.fail(jobs.log(job_id)[0])
    try:
        jobs.control(first['id'], 'pause')
        assert wait(first['id'], {'paused', 'failed'})['status'] == 'paused'
        with pytest.raises(ValueError, match='paused work'):
            jobs.start(first['request'])
        second = jobs.start({'kind': 'task', 'config': str(config), 'task': 'omp', 'action': 'run', 'conditions': [[5, 12]]})
        assert wait(second['id'], {'completed', 'failed'})['status'] == 'completed', jobs.log(second['id'])[0]
        assert jobs.task_job(config, 'omp', conditions=[[7, 8]])['id'] == first['id']
        assert recovery_path(config, 'omp', request=first['request']).exists()
        jobs.resume(first['id'])
        assert wait(first['id'], {'completed', 'failed'})['status'] == 'completed', jobs.log(first['id'])[0]
        assert not list((tmp_path / 'omp/.recovery').glob('*.json'))
    finally:
        for record in jobs.records():
            if record['status'] in {'queued', 'running', 'paused'}:
                jobs.control(record['id'], 'stop')
        for child in jobs.children.values():
            child.wait(timeout=10)


def test_cleanup_is_owned_by_one_condition_and_preserves_unrelated_progress(tmp_path):
    from cgap_sbl_jadce.checkpoints import Checkpoint, prune_owned_checkpoints
    old = {'definition': {'method': 'MMV-OMP', 'revision': 1}, 'condition': [25, 8]}
    other = {'definition': {'method': 'MMV-OMP', 'revision': 1}, 'condition': [30, 12]}
    first = Checkpoint(tmp_path, old)
    second = Checkpoint(tmp_path, other)
    first.save('0000', {'done': True})
    second.save('0000', {'done': True})
    unrelated = (second.folder / '0000.npz').read_bytes()
    prune_owned_checkpoints(tmp_path / '.checkpoints', {**old, 'gamma_count': 5})
    assert not (first.folder / '0000.npz').exists()
    assert (second.folder / '0000.npz').read_bytes() == unrelated
