"""Completion skips finished solvers; only an explicit rerun replaces their data."""
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.experiments import estimators
from cgap_sbl_jadce.gui.algorithm_actions import completion_plan
from cgap_sbl_jadce.gui.jobs import Jobs
from cgap_sbl_jadce.io import save_yaml
from cgap_sbl_jadce.paths import run_path
from cgap_sbl_jadce.recovery import begin_recovery, load_recoveries


class Failure(RuntimeError):
    pass


def wait(jobs, record):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = next(item for item in jobs.records() if item['id'] == record['id'])
        if state['status'] not in {'queued', 'running'}:
            return state
        time.sleep(.05)
    pytest.fail(jobs.log(record['id'])[0])


def test_plan_changes_from_start_to_complete_and_then_rerun(saved_data, tmp_path):
    config = tmp_path / 'input.yaml'
    save_yaml(config, saved_data)
    plan = completion_plan(config, 'omp')
    assert plan.label() == 'Start' and len(plan.start) == 3
    estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[[7, 8]])
    plan = completion_plan(config, 'omp')
    assert plan.label() == 'Complete all'
    assert plan.summary == '1 completed · 0 to resume · 2 to start'
    assert plan.selected([[7, 8]]).label(selected=True) == 'Rerun selected'
    assert plan.selected([[5, 12]]).label(selected=True) == 'Start'


def test_complete_all_skips_completed_files_and_preserves_measurements(saved_data, tmp_path):
    config = tmp_path / 'input.yaml'
    save_yaml(config, saved_data)
    estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[[7, 8]])
    folder = run_path(saved_data, 'omp', condition=(7, 8))
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in folder.rglob('*') if path.is_file()}
    jobs = Jobs(tmp_path)
    record = jobs.complete({'kind': 'task', 'config': str(config), 'task': 'omp', 'action': 'run'})
    try:
        state = wait(jobs, record)
        assert state['status'] == 'completed', jobs.log(record['id'])[0]
        assert record['complete']
        assert completion_plan(config, 'omp').label() == 'Rerun all'
        assert all(path.read_bytes() == data and path.stat().st_mtime_ns == stamp for path, (data, stamp) in before.items())
        # An idempotent completion request must also touch none of the final condition files.
        all_before = {path: path.stat().st_mtime_ns for path in run_path(saved_data, 'omp').rglob('*') if path.is_file()}
        record = jobs.complete(record['request'])
        assert wait(jobs, record)['status'] == 'completed', jobs.log(record['id'])[0]
        assert all(path.stat().st_mtime_ns == stamp for path, stamp in all_before.items())
    finally:
        if jobs.busy():
            jobs.control(jobs.busy()['id'], 'stop')
        for child in jobs.children.values():
            child.wait(timeout=10)


def test_complete_selected_reuses_units_from_an_interrupted_all_conditions_request(saved_data, tmp_path, monkeypatch):
    config = tmp_path / 'input.yaml'
    save_yaml(config, saved_data)
    original_request = {'kind': 'task', 'config': str(config), 'task': 'omp', 'action': 'run'}
    begin_recovery(original_request)
    spec = estimators.ESTIMATORS['omp']
    calls = []
    def solve(*args, **kwargs):
        calls.append(1)
        if len(calls) == 4:
            raise Failure()
        return spec.solve(*args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=solve))
    with pytest.raises(Failure):
        estimators.run_estimator_stage(saved_data, 'omp', False)
    folder = run_path(saved_data, 'omp', condition=(7, 12))
    checkpoint = next(folder.glob('.checkpoints/*/0000.npz'))
    with np.load(checkpoint) as archive:
        committed = json.loads(str(archive['_metadata']))['row']
    plan = completion_plan(config, 'omp')
    assert (7, 12) in plan.resume and (7, 8) in plan.completed
    jobs = Jobs(tmp_path)
    record = jobs.complete({**original_request, 'conditions': [[7, 12]]})
    try:
        assert wait(jobs, record)['status'] == 'completed', jobs.log(record['id'])[0]
        import pandas as pd
        data = pd.read_csv(folder / 'per_sample.csv')
        np.testing.assert_allclose(data[data.q.eq(0)].runtime_s, committed['runtime_s'])
        assert not run_path(saved_data, 'omp', 'summary.json', (5, 12)).exists()
        assert load_recoveries(config, 'omp')  # The original request still has one unfinished condition.
        record = jobs.complete(original_request)
        assert wait(jobs, record)['status'] == 'completed', jobs.log(record['id'])[0]
        assert not load_recoveries(config, 'omp')
    finally:
        if jobs.busy():
            jobs.control(jobs.busy()['id'], 'stop')
        for child in jobs.children.values():
            child.wait(timeout=10)


def test_incompatible_progress_blocks_completion_before_launch(saved_data, tmp_path, monkeypatch):
    config = tmp_path / 'input.yaml'
    save_yaml(config, saved_data)
    request = {'kind': 'task', 'config': str(config), 'task': 'omp', 'action': 'run', 'conditions': [[7, 8]]}
    begin_recovery(request)
    spec = estimators.ESTIMATORS['omp']
    calls = []
    def solve(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise Failure()
        return spec.solve(*args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=solve))
    with pytest.raises(Failure):
        estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[[7, 8]])
    manifest = next(run_path(saved_data, 'omp', condition=(7, 8)).glob('.checkpoints/*/definition.json'))
    manifest.write_text(manifest.read_text().replace('"machine":', '"original_machine":'))
    jobs = Jobs(tmp_path)
    assert (7, 8) in completion_plan(config, 'omp').blocked
    with pytest.raises(ValueError, match='explicitly Rerun'):
        jobs.complete(request)
    assert not jobs.records()
    assert list(run_path(saved_data, 'omp').rglob('.checkpoints/*/*.npz'))


def test_complete_all_adopts_a_paused_request_without_repeating_saved_work(saved_data, tmp_path):
    config = tmp_path / 'input.yaml'
    save_yaml(config, saved_data)
    jobs = Jobs(tmp_path)
    first = jobs.start({'kind': 'task', 'config': str(config), 'task': 'omp', 'action': 'run', 'conditions': [[7, 8]]})
    try:
        folder = run_path(saved_data, 'omp', condition=(7, 8))
        deadline = time.monotonic() + 30
        while not list(folder.glob('.checkpoints/*/0000.npz')):
            if time.monotonic() > deadline:
                pytest.fail('No saved solver unit available to pause')
            time.sleep(.005)
        jobs.control(first['id'], 'pause')
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            record = next(item for item in jobs.records() if item['id'] == first['id'])
            if record['status'] == 'paused':
                break
            time.sleep(.05)
        assert record['status'] == 'paused'
        with np.load(next(folder.glob('.checkpoints/*/0000.npz'))) as archive:
            committed = json.loads(str(archive['_metadata']))['row']
        second = jobs.complete({key: value for key, value in first['request'].items() if key != 'conditions'})
        assert second['id'] != first['id']
        assert wait(jobs, second)['status'] == 'completed', jobs.log(second['id'])[0]
        import pandas as pd
        data = pd.read_csv(folder / 'per_sample.csv')
        np.testing.assert_allclose(data[data.q.eq(0)].runtime_s, committed['runtime_s'])
        assert not load_recoveries(config, 'omp')
        assert jobs.task_job(config, 'omp')['status'] == 'completed'
    finally:
        for record in jobs.records():
            if record['status'] in {'running', 'queued', 'paused'}:
                jobs.control(record['id'], 'stop')
        for child in jobs.children.values():
            child.wait(timeout=10)


def test_gui_labels_and_complete_all_option_follow_selected_scope(saved_data, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    config = tmp_path / 'input.yaml'
    save_yaml(config, saved_data)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    source = Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py'
    app = AppTest.from_file(source, default_timeout=30).run()
    app.button(key='open_algorithm_omp').click().run()
    assert not app.exception
    assert app.button(key='task_action').label == 'Start'
    assert not app.button(key='complete_all').disabled
    estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[[7, 8]])
    app.run()
    assert app.button(key='task_action').label == 'Complete all'
    next(widget for widget in app.radio if widget.label == 'Condition scope').set_value('Selected conditions').run()
    assert app.button(key='task_action').label == 'Rerun selected'
    assert not app.button(key='complete_all').disabled
    selected = next(widget for widget in app.multiselect if widget.label == 'L / SNR conditions')
    selected.set_value([(5, 12)]).run()
    assert app.button(key='task_action').label == 'Start'
    assert app.button(key='complete_all')
    # Click Complete all and capture the request, without launching production work.
    captured = []
    monkeypatch.setattr(Jobs, 'complete', lambda self, request: captured.append(request))
    selected = next(widget for widget in app.multiselect if widget.label == 'L / SNR conditions')
    selected.set_value([(7, 8), (5, 12)]).run()
    assert app.button(key='task_action').label == 'Complete selected'
    assert app.button(key='rerun_conditions').label == 'Rerun selected'
    app.button(key='task_action').click().run()
    assert captured[0]['conditions'] == [[7, 8], [5, 12]]
    app.button(key='complete_all').click().run()
    assert not app.exception
    assert captured[-1]['task'] == 'omp' and 'conditions' not in captured[-1]
    assert not Jobs(tmp_path).records()


def test_missing_output_from_finished_condition_requires_explicit_rerun(saved_data, tmp_path):
    estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[[7, 8]])
    config = tmp_path / 'input.yaml'
    save_yaml(config, saved_data)
    request = {'kind': 'task', 'task': 'omp', 'action': 'run', 'config': str(config), 'conditions': [[7, 8]]}
    begin_recovery(request)
    run_path(saved_data, 'omp', 'figures/realization_scores.png', (7, 8)).unlink()
    plan = completion_plan(config, 'omp')
    assert (7, 8) in plan.blocked
    jobs = Jobs(tmp_path)
    with pytest.raises(ValueError, match='explicitly Rerun'):
        jobs.complete(request)
    assert not jobs.records()


def test_complete_all_progress_controls_remain_visible_in_selected_view(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from cgap_sbl_jadce.io import save_json
    config = tmp_path / 'input.yaml'
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path)
    save_yaml(config, cfg)
    jobs = Jobs(tmp_path)
    folder = jobs.folder / 'all-conditions'
    folder.mkdir()
    save_json(folder / 'control.json', {'command': 'run'})
    record = {'id': 'all-conditions', 'status': 'running',
              'request': {'kind': 'task', 'task': 'omp', 'action': 'run', 'config': str(config)}}
    monkeypatch.setattr(Jobs, 'task_job', lambda self, *a, conditions=None, **k: None if conditions is not None else record)
    monkeypatch.setattr(Jobs, 'busy', lambda self: record)
    monkeypatch.setattr(Jobs, 'log', lambda self, job_id: ('Working', None))
    calls = []
    monkeypatch.setattr(Jobs, 'control', lambda self, job_id, command: calls.append((job_id, command)))
    app = AppTest.from_string(f"""
import streamlit as st
from cgap_sbl_jadce.gui.jobs import Jobs
from cgap_sbl_jadce.gui.task_page import worker_panel
worker_panel(st, Jobs({str(tmp_path)!r}), {str(config)!r}, 'omp', 'run', [[7,8]])
""").run()
    assert not app.exception
    assert any(caption.value == 'Running scope: All conditions' for caption in app.caption)
    pause = next(button for button in app.button if button.label == 'Pause')
    assert not pause.disabled
    pause.click().run()
    assert calls == [('all-conditions', 'pause')]
