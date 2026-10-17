from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cgap_sbl_jadce.experiments import estimators
from cgap_sbl_jadce.experiments import spatial_tuning as gl_optuna_tuning
from cgap_sbl_jadce.io import save_yaml
from cgap_sbl_jadce.paths import tuning_path, task_dir, run_path
from cgap_sbl_jadce.settings import grouped_config, request_settings, validate_gl_search
from cgap_sbl_jadce.tasks import run_task


def settings(cfg):
    cfg = grouped_config(cfg)
    cfg['algorithms']['gl']['tuning'].update(trials=3, samples=2, startup_trials=2,
        tau_start=0., tau_stop=.5, tau_step=.1, selection_mode='per_condition')
    return cfg


def solver(calls, fail_at=None):
    def solve(r, C, parameters, cfg, **kwargs):
        assert set(parameters) == {'alpha'}
        assert C is not None
        if fail_at is not None and len(calls) == fail_at:
            raise RuntimeError('interrupted GL solver')
        calls.append((parameters['alpha'], r.Y.tobytes()))
        return SimpleNamespace(gamma=np.where(r.a, .2 + .05 * parameters['alpha'], .02),
                               mu=r.Z * (.9 - .01 * parameters['alpha']), iterations=2)
    return solve


def test_gl_tuning_run_comparison_convergence_and_all_exports(saved_data, tmp_path):
    from cgap_sbl_jadce.gui.charts import interactive_figures
    from cgap_sbl_jadce.gui.previews import render_preview, missing_preview_inputs
    from cgap_sbl_jadce.gui.workflow import numerical_state
    from cgap_sbl_jadce.gui.algorithm_actions import completion_plan
    from cgap_sbl_jadce.experiments.readiness import blocking_reasons
    cfg = settings(saved_data)
    pair = (7, 12.)
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    assert blocking_reasons(cfg, 'gl', 'tune', conditions=[pair]) == []
    assert blocking_reasons(cfg, 'gl', 'run', conditions=[pair])
    paths = run_task('gl', cfg, action='tune', conditions=[pair], show_progress=False)
    selected = estimators.load_selection(cfg, 'gl', pair)
    assert set(selected['parameters']) == {'alpha', 'tau'}
    assert selected['definition']['method'] == 'CGAP-SBL-GL'
    assert selected['definition']['samples'] == 2
    assert selected['search_method'] == 'optuna_tpe'
    assert len(paths) == 9 and all(path.exists() for path in paths)
    assert not tuning_path(cfg, 'glc', 'selected.json', pair).exists()
    assert completion_plan(config, 'gl', [pair], action='tune').completed == [pair]
    tune_cfg = estimators.condition_config(cfg, pair, 'gl')
    trials = pd.read_csv(tuning_path(tune_cfg, 'gl', 'trials.csv'))
    assert len(trials) == 3 and 'beta' not in trials
    assert trials.iloc[0].alpha == 0
    request = {'config': str(config), 'category': 'gl_tune', 'condition': list(pair)}
    figures = interactive_figures(request)
    assert [label for label, _ in figures] == ['TPE search history', 'Sampled alpha and threshold', 'Threshold scores',
        'Row norm threshold scores', 'Effective-channel energy · tuning realization 0']
    np.testing.assert_allclose(figures[1][1].data[0].x, trials.alpha)
    assert 'Selected α=' in figures[0][1].layout.title.text
    assert 'β' not in figures[0][1].layout.title.text
    assert missing_preview_inputs(tune_cfg, 'gl_tune') == []
    exports = render_preview(request, tmp_path / 'exports')
    assert len(exports) == 10 and {path.suffix for path in exports} == {'.png', '.pdf'}
    for path in exports:
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n' if path.suffix == '.png' else b'%PDF-')
    sweep_request = {'config': str(config), 'category': 'gl_tuning_sweeps'}
    assert len(interactive_figures(sweep_request)) == 4
    assert len(render_preview(sweep_request, tmp_path / 'sweep-exports')) == 8
    paths = run_task('gl', cfg, conditions=[pair], show_progress=False)
    assert paths and all(path.exists() for path in paths)
    assert 'gl' in estimators.completed_methods(cfg, [pair])
    assert 'gl' in estimators.convergence_methods(cfg, pair)
    scores = pd.read_csv(run_path(cfg, 'gl', 'per_sample.csv', pair))
    assert scores.algorithm.eq('CGAP-SBL-GL').all()
    assert {'nmse', 'runtime_s', 'f1'} <= set(scores)
    run_task('comparison', cfg, conditions=[pair], methods=['gl'], show_progress=False)
    run_task('convergence', cfg, condition=pair, methods=['gl'], show_progress=False)
    assert numerical_state(cfg, 'comparison').status == 'completed'
    with np.load(task_dir(cfg['run']['output_dir'], 'convergence') / 'convergence.npz') as data:
        assert {'gl_gamma_history', 'gl_lambda_history', 'gl_nmse_history'} <= set(data.files)
    for category in ['gl_run', 'comparison', 'convergence']:
        request = {'config': str(config), 'category': category}
        assert interactive_figures(request)
        assert render_preview(request, tmp_path / (category + '-exports'))


@pytest.mark.parametrize('boundary', ['solver', 'tell'])
def test_gl_resume_keeps_trial_sequence_and_completed_realizations(saved_data, tmp_path, monkeypatch, boundary):
    cfg = settings(saved_data)
    pair = (7, 12.)
    expected_cfg = deepcopy(cfg)
    expected_cfg['run']['output_dir'] = str(tmp_path / 'independent')
    for task in ['activity', 'communication', 'spatial_dependence']:
        shutil.copytree(task_dir(cfg['run']['output_dir'], task), task_dir(expected_cfg['run']['output_dir'], task))
    calls = []
    spec = estimators.ESTIMATORS['gl']
    monkeypatch.setitem(estimators.ESTIMATORS, 'gl', replace(spec, solve=solver(calls, 3 if boundary == 'solver' else None)))
    snapshot = gl_optuna_tuning._snapshot
    if boundary == 'tell':
        def interrupted(checkpoint, study, pending):
            if pending is None and sum(trial.state.is_finished() for trial in study.trials) == 1:
                raise RuntimeError('interrupted after GL tell')
            snapshot(checkpoint, study, pending)
        monkeypatch.setattr(gl_optuna_tuning, '_snapshot', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        run_task('gl', cfg, action='tune', conditions=[pair], show_progress=False)
    monkeypatch.setattr(gl_optuna_tuning, '_snapshot', snapshot)
    monkeypatch.setitem(estimators.ESTIMATORS, 'gl', replace(spec, solve=solver(calls)))
    run_task('gl', cfg, action='tune', conditions=[pair], show_progress=False, resume=True)
    assert len(calls) == 6 and len(set(calls)) == 6
    monkeypatch.setitem(estimators.ESTIMATORS, 'gl', replace(spec, solve=solver([])))
    run_task('gl', expected_cfg, action='tune', conditions=[pair], show_progress=False)
    assert estimators.load_selection(cfg, 'gl', pair)['parameters'] == estimators.load_selection(expected_cfg, 'gl', pair)['parameters']
    first, second = [pd.read_csv(tuning_path(item, 'gl', 'trials.csv', pair)).drop(columns='solver_runtime_s')
                     for item in (cfg, expected_cfg)]
    pd.testing.assert_frame_equal(first, second)
    assert not list(task_dir(cfg['run']['output_dir'], 'gl').rglob('.checkpoints/*/*.npz'))


def test_gl_settings_and_scoped_reruns_preserve_glc_and_other_conditions(saved_data, monkeypatch):
    from cgap_sbl_jadce.gui.algorithm_actions import completion_plan
    from cgap_sbl_jadce.additional_conditions import add_condition
    cfg = settings(saved_data)
    glc_definition = estimators.tuning_definition(cfg, 'glc')
    glc_settings = request_settings(cfg, {'task': 'glc', 'action': 'tune'})
    cfg['algorithms']['gl']['tuning']['alpha_high'] = 25.
    assert estimators.tuning_definition(cfg, 'glc') == glc_definition
    assert request_settings(cfg, {'task': 'glc', 'action': 'tune'}) == glc_settings
    glc_path = tuning_path(cfg, 'glc', 'untouched.txt', (7, 12.))
    glc_path.parent.mkdir(parents=True)
    glc_path.write_text('GLC results are independently owned')
    calls = []
    monkeypatch.setitem(estimators.ESTIMATORS, 'gl', replace(estimators.ESTIMATORS['gl'], solve=solver(calls)))
    pairs = [(7, 12.), (5, 12.)]
    run_task('gl', cfg, action='tune', conditions=pairs, show_progress=False)
    saved = tuning_path(cfg, 'gl', 'selected.json', pairs[1]).read_bytes()
    run_task('gl', cfg, action='tune', conditions=[pairs[0]], show_progress=False)
    assert tuning_path(cfg, 'gl', 'selected.json', pairs[1]).read_bytes() == saved
    assert glc_path.read_text() == 'GLC results are independently owned'
    assert len(calls) == 18
    config = Path(cfg['run']['output_dir']) / 'source.yaml'
    save_yaml(config, cfg)
    plan = completion_plan(config, 'gl', pairs, action='tune')
    assert plan.completed == pairs and not plan.blocked
    add_condition(cfg, 'gl', (9, 10.))
    assert (9, 10.) in estimators.tuning_conditions(cfg, [(9, 10.)], 'gl')
    assert (9, 10.) not in estimators.tuning_conditions(cfg, method='gl')


def test_gl_gui_has_independent_settings_and_condition_controls(saved_data, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    cfg = settings(saved_data)
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    assert not app.exception
    assert app.button(key='open_algorithm_glc') and app.button(key='open_algorithm_gl')
    app.button(key='open_algorithm_gl').click().run()
    app.button(key='open_gl_tune').click().run()
    assert not app.exception
    labels = [widget.label for widget in app.number_input]
    assert 'Alpha high' in labels and 'Beta high' not in labels
    assert any(widget.label == 'Condition scope' for widget in app.radio)
    assert any('TPE: 3 trials × 2' in widget.value for widget in app.caption)
    assert any(button.label == 'Add' for button in app.button)


@pytest.mark.parametrize('updates', [{'alpha_high': -1}, {'alpha_low': float('nan')}, {'samples': 3},
    {'trials': 0}, {'startup_trials': 8}, {'optuna_seed': -2}, {'beta_low': 0.}])
def test_gl_validates_independent_search_settings(saved_data, updates):
    cfg = settings(saved_data)
    cfg['algorithms']['gl']['tuning'].update(updates)
    with pytest.raises(ValueError, match='GL'):
        validate_gl_search(cfg)
