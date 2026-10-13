from dataclasses import replace
from pathlib import Path
import runpy

import pytest

from cgap_sbl_jadce import tasks
from cgap_sbl_jadce.config import load_config


@pytest.mark.parametrize('task_id', sorted(tasks.TASKS))
def test_action_runs_only_its_stage_then_regenerates_png(tmp_path, monkeypatch, task_id):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    called = []
    def stage(*args, **kwargs):
        called.append('stage')
    def render(cfg, **kwargs):
        called.append(kwargs)
        return [tmp_path / 'figure.png']
    monkeypatch.setitem(tasks.TASKS, task_id, replace(tasks.TASKS[task_id], stage=stage, plots=(render,)))
    assert tasks.run_task(task_id, cfg) == [tmp_path / 'figure.png']
    assert called == ['stage', {'formats': 'png'}]


def test_invalid_action_never_runs_a_stage(monkeypatch):
    monkeypatch.setitem(tasks.TASKS, 'omp', replace(tasks.TASKS['omp'], stage=lambda *a, **k: pytest.fail('Ran invalid action')))
    with pytest.raises(ValueError, match='no tune'):
        tasks.run_task('omp', {}, action='tune')


def test_conditions_passed_to_only_selected_algorithm(monkeypatch):
    calls = []
    monkeypatch.setitem(tasks.TASKS, 'cosamp', replace(tasks.TASKS['cosamp'], stage=lambda cfg, **kwargs: calls.append(kwargs), plots=()))
    tasks.run_task('cosamp', {}, conditions=[[25, 8]], show_progress=False)
    assert calls == [{'show_progress': False, 'conditions': [[25, 8]]}]


def test_future_algorithm_exposes_tune_and_run(monkeypatch):
    spec = replace(tasks.estimators.ESTIMATORS['anc'], task='gl', name='CGAP-SBL-GL')
    monkeypatch.setitem(tasks.estimators.ESTIMATORS, 'gl', spec)
    monkeypatch.setitem(tasks.TASKS, 'gl', replace(tasks.TASKS['anc'], description='run CGAP-SBL-GL'))
    assert list(tasks.task_actions('gl')) == ['tune', 'run']
    assert list(tasks.task_actions('omp')) == ['run']


def test_imports_do_not_run_tasks_or_open_gui(monkeypatch):
    from cgap_sbl_jadce.gui import app
    monkeypatch.setattr(app, 'main', lambda *args: pytest.fail('Import opened GUI'))
    runpy.run_module('cgap_sbl_jadce.__main__', run_name='import_check')
    for script in Path('scripts').rglob('*.py'):
        runpy.run_path(str(script), run_name='import_check')
