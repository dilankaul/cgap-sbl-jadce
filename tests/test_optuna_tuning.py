from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cgap_sbl_jadce.experiments import estimators, optuna_tuning
from cgap_sbl_jadce.experiments.anc_grid_study import anc_grid_study_stage, grid_definition
from cgap_sbl_jadce.experiments.search import rank_score
from cgap_sbl_jadce.io import save_yaml
from cgap_sbl_jadce.paths import tuning_path, task_dir
from cgap_sbl_jadce.settings import grouped_config, request_settings


def settings(saved_data):
    cfg = grouped_config(saved_data)
    cfg['algorithms']['anc']['tuning'].update(trials=4, samples=2, startup_trials=2,
                                            tau_start=0., tau_stop=.5, tau_step=.1)
    cfg['anc_grid_study'].update(alpha_start=0., alpha_stop=2., alpha_step=2.,
                                 beta_start=0., beta_stop=.1, beta_step=.1,
                                 tau_start=0., tau_stop=.5, tau_step=.1)
    return cfg


def solver(calls, fail_at=None):
    def solve(r, C, parameters, cfg, **kwargs):
        key = (parameters['alpha'], parameters['beta'], r.Y.tobytes())
        if fail_at is not None and len(calls) == fail_at:
            raise RuntimeError('interrupted solver')
        calls.append(key)
        gamma = np.where(r.a, .2 + .1 * parameters['alpha'], .02 + parameters['beta'])
        return SimpleNamespace(gamma=gamma, mu=r.Z * (.9 - .01 * parameters['alpha']), iterations=2)
    return solve


def test_full_tpe_scores_every_trial_on_all_samples_and_selects_winner(saved_data, monkeypatch):
    cfg = settings(saved_data)
    calls = []
    monkeypatch.setitem(estimators.ESTIMATORS, 'anc', replace(estimators.ESTIMATORS['anc'], solve=solver(calls)))
    selected = estimators.tune_estimator_stage(cfg, 'anc', False)
    assert len(calls) == 4 * 2
    assert len(set(calls)) == len(calls)
    assert calls[0][:2] == (0., 0.)
    folder = tuning_path(cfg, 'anc', '')
    trials = pd.read_csv(folder / 'trials.csv')
    assert len(trials) == 4 and trials.samples.eq(2).all()
    samples = [{key[2] for key in calls[index:index + 2]} for index in range(0, len(calls), 2)]
    assert all(len(values) == 2 and values == samples[0] for values in samples)
    best = max(trials.to_dict('records'), key=rank_score)
    assert selected['parameters'] == pytest.approx({key: best[key] for key in ('alpha', 'beta', 'tau')})
    assert 'selection_stage' not in selected and selected['search_method'] == 'optuna_tpe'
    assert selected['definition']['samples'] == 2
    assert selected['definition']['sample_indices'] == [0, 1]
    assert json.loads((folder / 'search_progress.json').read_text()) == {'completed': 4, 'total': 4, 'samples': 2}
    assert not (folder / 'finalists.csv').exists()
    assert estimators.load_selection(cfg, 'anc') == selected
    assert (folder / 'study.sqlite3').exists() and (folder / 'sampler.pkl').exists()
    assert not list(folder.rglob('.checkpoints/*/*.npz'))
    assert not (folder / 'alpha_beta.csv').exists()
    curve = pd.read_csv(folder / 'thresholds.csv')
    threshold = max(curve.to_dict('records'), key=lambda row: (row['f1'], row['recall'], row['precision']))
    assert threshold['tau'] == pytest.approx(selected['parameters']['tau'])
    assert threshold['f1'] == pytest.approx(selected['tuning_scores']['f1'])
    import optuna
    storage = optuna.storages.RDBStorage('sqlite:///' + str((folder / 'study.sqlite3').resolve()))
    try:
        study = optuna.load_study(study_name='anc_tpe', storage=storage)
        assert [trial.value for trial in study.trials] == pytest.approx(trials.f1.tolist())
        assert len(study.trials) == 4
    finally:
        storage.remove_session()
        storage.engine.dispose()
    from cgap_sbl_jadce.gui.charts import interactive_figures
    from cgap_sbl_jadce.gui.previews import missing_preview_inputs
    config = Path(cfg['run']['output_dir']) / 'source.yaml'
    save_yaml(config, cfg)
    assert missing_preview_inputs(cfg, 'anc_tune') == []
    figures = interactive_figures({'config': str(config), 'category': 'anc_tune'})
    assert [title for title, _ in figures] == ['TPE search history', 'Sampled parameter pairs', 'Threshold scores']
    np.testing.assert_allclose(figures[0][1].data[0].y, trials.f1)
    np.testing.assert_allclose(figures[1][1].data[0].marker.color, trials.f1)


def test_full_tpe_default_budget_and_validation():
    from cgap_sbl_jadce.config import load_config
    from cgap_sbl_jadce.settings import validate_anc_search
    for path in ('configs/experiment.yaml', 'results/experiment/config.yaml'):
        cfg = load_config(path)
        values = validate_anc_search(cfg)
        assert values['search_method'] == 'optuna_tpe'
        assert values['trials'] == 50 and values['samples'] == 100
        assert values['startup_trials'] == 10
        assert not any(key.startswith(('stage1_', 'stage2_')) for key in cfg['algorithms']['anc']['tuning'])


@pytest.mark.parametrize('updates, message', [
    ({'trials': 0}, 'positive integer'),
    ({'trials': 1.5}, 'positive integer'),
    ({'samples': 3}, 'saved tuning sample count'),
    ({'samples': 0}, 'positive integer'),
    ({'startup_trials': 5}, 'cannot exceed the trial count'),
    ({'search_method': 'optuna_tpe_two_stage'}, 'full Optuna TPE'),
])
def test_full_tpe_rejects_invalid_budgets(saved_data, updates, message):
    from cgap_sbl_jadce.settings import validate_anc_search
    cfg = settings(saved_data)
    cfg['algorithms']['anc']['tuning'].update(updates)
    with pytest.raises(ValueError, match=message):
        validate_anc_search(cfg)


@pytest.mark.parametrize('boundary', ['solver', 'tell'])
def test_resuming_reproduces_uninterrupted_trials_and_reuses_committed_work(saved_data, tmp_path, monkeypatch, boundary):
    cfg = settings(saved_data)
    independent = deepcopy(cfg)
    independent['run']['output_dir'] = str(tmp_path / 'uninterrupted')
    for task in ('activity', 'communication', 'spatial_dependence'):
        shutil.copytree(task_dir(cfg['run']['output_dir'], task), task_dir(independent['run']['output_dir'], task))
    spec = estimators.ESTIMATORS['anc']
    calls = []
    if boundary == 'solver':
        monkeypatch.setitem(estimators.ESTIMATORS, 'anc', replace(spec, solve=solver(calls, fail_at=3)))
    else:
        snapshot = optuna_tuning._snapshot
        def fail_after_tell(checkpoint, study, pending):
            if pending is None and sum(trial.state.is_finished() for trial in study.trials) == 1:
                raise RuntimeError('interrupted after tell')
            return snapshot(checkpoint, study, pending)
        monkeypatch.setattr(optuna_tuning, '_snapshot', fail_after_tell)
        monkeypatch.setitem(estimators.ESTIMATORS, 'anc', replace(spec, solve=solver(calls)))
    with pytest.raises(RuntimeError, match='interrupted'):
        estimators.tune_estimator_stage(cfg, 'anc', False)
    if boundary == 'tell':
        monkeypatch.setattr(optuna_tuning, '_snapshot', snapshot)
    monkeypatch.setitem(estimators.ESTIMATORS, 'anc', replace(spec, solve=solver(calls)))
    resumed = estimators.tune_estimator_stage(cfg, 'anc', False, resume=True)
    assert len(calls) == 8 and len(set(calls)) == 8
    monkeypatch.setitem(estimators.ESTIMATORS, 'anc', replace(spec, solve=solver([])))
    expected = estimators.tune_estimator_stage(independent, 'anc', False)
    assert resumed['parameters'] == expected['parameters']
    for filename in ('trials.csv', 'thresholds.csv'):
        first = pd.read_csv(tuning_path(cfg, 'anc', filename)).drop(columns='solver_runtime_s', errors='ignore')
        second = pd.read_csv(tuning_path(independent, 'anc', filename)).drop(columns='solver_runtime_s', errors='ignore')
        pd.testing.assert_frame_equal(first, second)


def test_paper_grid_is_independent_and_cannot_provide_algorithm_selection(saved_data, monkeypatch):
    cfg = settings(saved_data)
    calls = []
    monkeypatch.setitem(estimators.ESTIMATORS, 'anc', replace(estimators.ESTIMATORS['anc'], solve=solver(calls)))
    selected = estimators.tune_estimator_stage(cfg, 'anc', False)
    path = tuning_path(cfg, 'anc', 'selected.json')
    contents = path.read_bytes()
    original = estimators.tuning_definition(cfg, 'anc')
    cfg['anc_grid_study']['alpha_stop'] = 4.
    assert estimators.tuning_definition(cfg, 'anc') == original
    before_grid = grid_definition(cfg)
    cfg['algorithms']['anc']['tuning']['trials'] += 1
    assert grid_definition(cfg) == before_grid
    request = {'kind': 'task', 'task': 'anc', 'action': 'tune'}
    before_request = request_settings(cfg, request)
    cfg['anc_grid_study']['alpha_stop'] = 2.
    assert request_settings(cfg, request) == before_request
    summary = anc_grid_study_stage(cfg, False)
    assert summary['pairs'] == 4
    assert path.read_bytes() == contents
    folder = task_dir(cfg['run']['output_dir'], 'anc_grid_study')
    assert not (folder / 'selected.json').exists()
    assert len(pd.read_csv(folder / 'alpha_beta.csv')) == 4
    rejected = dict(selected)
    rejected['search_method'] = 'optuna_tpe_two_stage'
    rejected['selection_stage'] = 2
    path.write_text(json.dumps(rejected))
    with pytest.raises(ValueError, match='full Optuna TPE'):
        estimators.load_selection(cfg, 'anc')


def test_grid_study_gui_is_fixed_condition_and_cannot_run_estimators(saved_data, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from cgap_sbl_jadce.gui.jobs import Jobs
    cfg = settings(saved_data)
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    requests = []
    monkeypatch.setattr(Jobs, 'start', lambda self, request: requests.append(request))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    app.button(key='open_anc_grid_study_run').click().run()
    assert not app.exception
    assert not [widget for widget in app.radio if widget.label == 'Condition scope']
    assert len([widget for widget in app.number_input if widget.label == 'Grid study pilot length L']) == 1
    assert not app.button(key='task_action').disabled
    app.button(key='task_action').click().run()
    assert requests == [{'kind': 'task', 'task': 'anc_grid_study', 'action': 'run', 'config': str(config)}]
    assert not app.exception


def test_grid_worker_saves_separate_results_and_interactive_exports(saved_data, tmp_path):
    import time
    from cgap_sbl_jadce.gui.charts import interactive_figures
    from cgap_sbl_jadce.gui.jobs import Jobs
    from cgap_sbl_jadce.gui.workflow import numerical_state
    from cgap_sbl_jadce.gui.previews import render_preview
    cfg = settings(saved_data)
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    jobs = Jobs(tmp_path)
    record = jobs.start({'kind': 'task', 'task': 'anc_grid_study', 'action': 'run', 'config': str(config)})
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = next(item for item in jobs.records() if item['id'] == record['id'])
            if state['status'] not in {'queued', 'running'}:
                break
            time.sleep(.05)
        assert state['status'] == 'completed', jobs.log(record['id'])[0]
        assert len(state['paths']) == 3
        assert not tuning_path(cfg, 'anc', 'selected.json').exists()
        assert numerical_state(cfg, 'anc_grid_study').status == 'completed'
        request = {'config': str(config), 'category': 'anc_grid_study'}
        figures = interactive_figures(request)
        assert len(figures) == 3
        table = pd.read_csv(task_dir(cfg['run']['output_dir'], 'anc_grid_study') / 'alpha_beta.csv')
        expected = table.pivot(index='alpha', columns='beta', values='f1').sort_index().sort_index(axis=1)
        np.testing.assert_allclose(figures[0][1].data[0].z, expected.to_numpy())
        paths = render_preview(request, tmp_path / 'exports')
        assert len(paths) == 6 and {path.suffix for path in paths} == {'.png', '.pdf'}
        assert all(path.exists() for path in paths)
    finally:
        if jobs.busy():
            jobs.control(record['id'], 'stop')
        jobs.children[record['id']].wait(timeout=10)


@pytest.mark.parametrize('method', ['sbl', 'anc'])
def test_tuning_final_cleanup_recovers_without_repeating_completed_solvers(saved_data, tmp_path, monkeypatch, method):
    from cgap_sbl_jadce.checkpoints import Checkpoint
    from cgap_sbl_jadce.recovery import begin_recovery, finish_recovery
    from cgap_sbl_jadce.tasks import run_task
    cfg = settings(saved_data)
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    request = {'kind': 'task', 'task': method, 'action': 'tune', 'config': str(config)}
    begin_recovery(request)
    cfg['task_runtime'] = {'input_config': str(config)}
    clear = Checkpoint.clear
    def interrupted_cleanup(checkpoint, **kwargs):
        if checkpoint.units:
            raise RuntimeError('interrupted cleanup')
        return clear(checkpoint, **kwargs)
    monkeypatch.setattr(Checkpoint, 'clear', interrupted_cleanup)
    with pytest.raises(RuntimeError, match='interrupted cleanup'):
        run_task(method, cfg, action='tune', show_progress=False)
    monkeypatch.setattr(Checkpoint, 'clear', clear)
    spec = estimators.ESTIMATORS[method]
    monkeypatch.setitem(estimators.ESTIMATORS, method, replace(spec,
        solve=lambda *a, **k: pytest.fail('Repeated completed solver during final cleanup')))
    paths = run_task(method, cfg, action='tune', show_progress=False, resume=True)
    assert paths and all(path.exists() for path in paths)
    assert not list(tuning_path(cfg, method, '').rglob('.checkpoints/*/*.npz'))
    finish_recovery(request)


def test_structured_legacy_grid_values_are_retained_by_the_paper_study(saved_data):
    from cgap_sbl_jadce.settings import grid_study_settings, range_values
    cfg = deepcopy(saved_data)
    cfg['algorithms']['anc']['tuning'].update(alpha_start=1., alpha_stop=3., alpha_step=1.,
                                            beta_start=.02, beta_stop=.06, beta_step=.02)
    normalized = grouped_config(cfg)
    study = grid_study_settings(normalized)
    assert range_values(study, 'alpha') == [1., 2., 3.]
    assert range_values(study, 'beta') == [.02, .04, .06]
    assert normalized['algorithms']['anc']['tuning']['alpha_low'] == 0.
    assert 'alpha_start' not in normalized['algorithms']['anc']['tuning']
