from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest

from cgap_sbl_jadce.experiments import estimators
from cgap_sbl_jadce.gui.jobs import Jobs
from cgap_sbl_jadce.gui.tuning_conditions import tuning_plan
from cgap_sbl_jadce.io import save_yaml
from cgap_sbl_jadce.paths import tuning_path, tuning_figure_path
from cgap_sbl_jadce.settings import grouped_config, communication_conditions


def config_for(saved_data, method='anc'):
    cfg = grouped_config(saved_data)
    cfg['algorithms'][method]['tuning']['selection_mode'] = 'per_condition'
    cfg['algorithms']['anc']['tuning'].update(trials=2, samples=2, startup_trials=1)
    return cfg


def wait(jobs, record):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        state = next(item for item in jobs.records() if item['id'] == record['id'])
        if state['status'] not in {'running', 'queued'}:
            return state
        time.sleep(.05)
    pytest.fail(jobs.log(record['id'])[0])


def test_tune_and_communication_follow_run_sweep_order(saved_data, tmp_path):
    cfg = config_for(saved_data)
    path = tmp_path / 'source.yaml'
    save_yaml(path, cfg)
    expected = list(dict.fromkeys(value[:2] for value in estimators.evaluation_conditions(cfg)))
    assert tuning_plan(path, 'anc').pairs == expected
    assert communication_conditions(cfg) == expected
    from cgap_sbl_jadce.gui.communication_conditions import condition_rows
    rows, issues = condition_rows(cfg)
    assert not issues
    assert [(row['Pilot length L'], row['SNR (dB)']) for row in rows] == expected
    assert all(row['Type'] == 'Sweep' for row in rows)


def test_tune_complete_worker_preserves_completed_condition_and_finishes_the_rest(saved_data, tmp_path):
    cfg = config_for(saved_data)
    path = tmp_path / 'source.yaml'
    save_yaml(path, cfg)
    first = (7, 8.)
    estimators.tune_estimator_stage(cfg, 'anc', False, conditions=[first])
    folder = tuning_path(cfg, 'anc', '', first)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.rglob('*') if p.is_file()}
    jobs = Jobs(tmp_path)
    record = jobs.complete({'kind': 'task', 'config': str(path), 'task': 'anc', 'action': 'tune'})
    try:
        state = wait(jobs, record)
        assert state['status'] == 'completed', jobs.log(record['id'])[0]
        assert tuning_plan(path, 'anc').label() == 'Rerun all'
        assert all(p.read_bytes() == contents and p.stat().st_mtime_ns == modified
                   for p, (contents, modified) in before.items())
        assert len(state['paths']) == 3 * 3 + 6
    finally:
        if jobs.busy():
            jobs.control(record['id'], 'stop')
        for child in jobs.children.values():
            child.wait(timeout=10)


def test_tune_rerun_clears_selected_scope_before_solver_and_preserves_other_files(saved_data, monkeypatch):
    cfg = config_for(saved_data, 'sbl')
    estimators.tune_estimator_stage(cfg, 'sbl', False)
    scope = [(7, 8.), (7, 12.)]
    untouched = tuning_path(cfg, 'sbl', '', (5, 12.))
    before = {p: p.read_bytes() for p in untouched.rglob('*') if p.is_file()}
    sentinels = []
    for pair in scope:
        sentinel = tuning_path(cfg, 'sbl', 'notes.txt', pair)
        sentinel.write_text('keep this')
        sentinels.append(sentinel)
    def fail(*args, **kwargs):
        for pair in scope:
            assert not tuning_path(cfg, 'sbl', 'selected.json', pair).exists()
            assert not tuning_path(cfg, 'sbl', 'summary.json', pair).exists()
            assert not tuning_figure_path(estimators.condition_config(cfg, pair, 'sbl'), 'sbl', 'sbl_threshold_scores.png').exists()
        raise RuntimeError('solver interrupted')
    monkeypatch.setitem(estimators.ESTIMATORS, 'sbl', replace(estimators.ESTIMATORS['sbl'], solve=fail))
    with pytest.raises(RuntimeError, match='solver interrupted'):
        estimators.tune_estimator_stage(cfg, 'sbl', False, conditions=scope)
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert all(p.read_text() == 'keep this' for p in sentinels)


def test_tune_reset_interruption_is_recoverable_without_repeating_finished_reset(saved_data, tmp_path, monkeypatch):
    from cgap_sbl_jadce.recovery import begin_recovery
    from cgap_sbl_jadce.tuning_results import pending_tuning_resets
    cfg = config_for(saved_data, 'sbl')
    scope = [(7, 8.), (7, 12.)]
    estimators.tune_estimator_stage(cfg, 'sbl', False, conditions=scope)
    path = tmp_path / 'source.yaml'
    save_yaml(path, cfg)
    request = {'kind': 'task', 'config': str(path), 'task': 'sbl', 'action': 'tune',
               'conditions': [list(pair) for pair in scope]}
    begin_recovery(request, reset=True)
    blocked = tuning_path(cfg, 'sbl', 'selected.json', scope[1])
    unlink = Path.unlink
    def fail_unlink(p, *args, **kwargs):
        if p == blocked:
            raise RuntimeError('interrupted reset')
        return unlink(p, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', fail_unlink)
    with pytest.raises(RuntimeError, match='interrupted reset'):
        estimators.tune_estimator_stage(cfg, 'sbl', False, conditions=scope)
    assert pending_tuning_resets(cfg, 'sbl') == set(scope)
    assert set(tuning_plan(path, 'sbl', scope).blocked) == set(scope)
    with pytest.raises(ValueError, match='interrupted tuning reset'):
        estimators.load_selection(cfg, 'sbl', scope[1])
    monkeypatch.setattr(Path, 'unlink', unlink)
    original = estimators.ESTIMATORS['sbl']
    calls = []
    def interrupted(*args, **kwargs):
        if len(calls) == 1:
            raise RuntimeError('interrupted solver')
        calls.append(1)
        return original.solve(*args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'sbl', replace(original, solve=interrupted))
    with pytest.raises(RuntimeError, match='interrupted solver'):
        estimators.tune_estimator_stage(cfg, 'sbl', False, conditions=scope, resume=True)
    assert not pending_tuning_resets(cfg, 'sbl')
    monkeypatch.setitem(estimators.ESTIMATORS, 'sbl', replace(original, solve=lambda *a, **k: (calls.append(1), original.solve(*a, **k))[1]))
    estimators.tune_estimator_stage(cfg, 'sbl', False, conditions=scope, resume=True)
    assert len(calls) == 4
    assert len(tuning_plan(path, 'sbl', scope).completed) == 2


def test_parameter_sweep_exports_use_saved_selections_and_keep_existing_plots(saved_data, tmp_path, monkeypatch):
    from cgap_sbl_jadce.figures.estimator_parameter_sweeps import parameter_sweeps, plot_tuning_parameter_sweeps
    from cgap_sbl_jadce.gui.charts import interactive_figures
    from cgap_sbl_jadce.gui.previews import render_preview
    cfg = config_for(saved_data)
    estimators.tune_estimator_stage(cfg, 'anc', False, conditions=[(7, 8.), (5, 12.)])
    path = tmp_path / 'source.yaml'
    save_yaml(path, cfg)
    monkeypatch.setitem(estimators.ESTIMATORS, 'anc', replace(estimators.ESTIMATORS['anc'], solve=lambda *a, **k: pytest.fail('Plotting ran solver')))
    curves = parameter_sweeps(cfg, 'anc')
    assert len(curves) == 6
    selected = estimators.load_selection(cfg, 'anc', (7, 8.))['parameters']
    for curve in curves[:3]:
        assert curve['x'] == [8., 12.]
        assert curve['y'][0] == selected[curve['parameter']]
        assert np.isnan(curve['y'][1])
    request = {'config': str(path), 'category': 'anc_tuning_sweeps'}
    figures = interactive_figures(request)
    assert len(figures) == 6
    for (_, figure), curve in zip(figures, curves):
        np.testing.assert_allclose(figure.data[0].y, curve['y'], equal_nan=True)
        assert not figure.data[0].connectgaps
        assert figure.layout.xaxis.range == tuple(curve['x'])
    outputs = render_preview(request, tmp_path / 'sweep_exports')
    assert len(outputs) == 12 and {p.suffix for p in outputs} == {'.png', '.pdf'}
    assert all(p.exists() and p.stat().st_size > 100 for p in outputs)
    request.update(category='anc_tune', condition=[7, 8.])
    diagnostics = interactive_figures(request)
    assert len(diagnostics) == 3
    assert all('Selected α=' in figure.layout.title.text and 'β=' in figure.layout.title.text for _, figure in diagnostics)
    outputs = render_preview(request, tmp_path / 'diagnostic_exports')
    assert len(outputs) == 6 and all(p.exists() for p in outputs)
    cfg['algorithms']['anc']['solver']['delta'] *= 2
    assert parameter_sweeps(cfg, 'anc') == []


def test_stopped_tune_shows_resume_and_rerun_selected(saved_data, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from cgap_sbl_jadce.gui.jobs import config_digest
    from cgap_sbl_jadce.io import save_json
    from cgap_sbl_jadce.recovery import begin_recovery
    cfg = config_for(saved_data, 'sbl')
    path = tmp_path / 'source.yaml'
    save_yaml(path, cfg)
    request = {'kind': 'task', 'config': str(path), 'task': 'sbl', 'action': 'tune', 'conditions': [[7, 8.]]}
    begin_recovery(request)
    record = {'id': 'stopped', 'request': request, 'status': 'stopped', 'created': time.time(),
              'updated': time.time(), 'config_digest': config_digest(path, request)}
    save_json(tmp_path / '.gui/stopped/job.json', record)
    save_json(tmp_path / '.gui/stopped/control.json', {'command': 'stop'})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(path))
    calls = []
    monkeypatch.setattr(Jobs, 'start', lambda self, request, **kwargs: calls.append((request, kwargs)))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    app.button(key='open_algorithm_sbl').click().run()
    app.button(key='open_sbl_tune').click().run()
    next(widget for widget in app.radio if widget.label == 'Condition scope').set_value('Selected conditions').run()
    assert not app.exception
    assert app.button(key='task_action').label == 'Resume'
    assert app.button(key='rerun_conditions').label == 'Rerun selected'
    assert not app.button(key='rerun_conditions').disabled
    table = next(widget for widget in app.expander if widget.label == 'Condition status')
    assert sum(widget.value == 'Sweep' for widget in table.markdown) == 3
    assert not any(widget.value == 'Setup' for widget in table.markdown)
    app.button(key='rerun_conditions').click().run()
    assert not app.exception
    assert calls == [(request, {})]


def test_linked_tuning_scope_blocks_reset_without_touching_completed_results(saved_data, tmp_path):
    cfg = config_for(saved_data, 'sbl')
    estimators.tune_estimator_stage(cfg, 'sbl', False, conditions=[(7, 8.)])
    path = tuning_path(cfg, 'sbl', 'selected.json', (7, 8.))
    contents = path.read_bytes()
    external = tmp_path / 'outside'
    external.mkdir()
    tuning_path(cfg, 'sbl', '', (7, 12.)).symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='outside'):
        estimators.tune_estimator_stage(cfg, 'sbl', False, conditions=[(7, 8.), (7, 12.)])
    assert path.read_bytes() == contents
    assert not list(external.iterdir())


def test_sbl_parameter_sweeps_keep_its_own_thresholds(saved_data):
    from cgap_sbl_jadce.figures.estimator_parameter_sweeps import parameter_sweeps
    cfg = config_for(saved_data, 'sbl')
    estimators.tune_estimator_stage(cfg, 'sbl', False, conditions=[(7, 8.), (5, 12.)])
    curves = parameter_sweeps(cfg, 'sbl')
    assert len(curves) == 2 and all(curve['parameter'] == 'tau' for curve in curves)
    assert curves[0]['y'][0] == estimators.load_selection(cfg, 'sbl', (7, 8.))['parameters']['tau']
    assert curves[1]['y'][0] == estimators.load_selection(cfg, 'sbl', (5, 12.))['parameters']['tau']
