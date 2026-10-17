from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cgap_sbl_jadce.experiments import estimators
from cgap_sbl_jadce.io import save_yaml
from cgap_sbl_jadce.paths import task_dir, tuning_path, run_path
from cgap_sbl_jadce.settings import grouped_config
from cgap_sbl_jadce.tasks import run_task
from cgap_sbl_jadce.variants import parameters

NEW_METHODS = ('nglc', 'rgl', 'gd', 'aga')


def setup(cfg, method):
    cfg = grouped_config(cfg)
    cfg['algorithms'][method]['tuning'].update(trials=2, startup_trials=1,
        selection_mode='per_condition', tau_start=0., tau_stop=.5, tau_step=.1)
    return cfg


@pytest.mark.parametrize('method', NEW_METHODS)
def test_variant_tuning_evaluation_convergence_and_saved_data_plots(saved_data, tmp_path, method):
    from cgap_sbl_jadce.gui.algorithm_actions import completion_plan
    from cgap_sbl_jadce.gui.charts import interactive_figures
    from cgap_sbl_jadce.gui.previews import render_preview
    from cgap_sbl_jadce.experiments.readiness import blocking_reasons
    cfg = setup(saved_data, method)
    pair = (7, 12.)
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    assert not blocking_reasons(cfg, method, 'tune', conditions=[pair])
    assert blocking_reasons(cfg, method, 'run', conditions=[pair])
    paths = run_task(method, cfg, action='tune', conditions=[pair], show_progress=False)
    assert paths and all(p.exists() for p in paths)
    selection = estimators.load_selection(cfg, method, pair)
    assert set(selection['parameters']) == {*parameters(method), 'tau'}
    assert selection['definition']['method'] == 'CGAP-SBL-' + method.upper()
    assert selection['definition']['sample_indices'] == [0, 1]
    assert not blocking_reasons(cfg, method, 'run', conditions=[pair])
    assert completion_plan(config, method, [pair], action='tune').completed == [pair]
    trials = pd.read_csv(tuning_path(cfg, method, 'trials.csv', pair))
    assert trials.iloc[0].alpha == 0 and len(trials) == 2
    assert {*parameters(method), 'tau'} <= set(trials)
    norm_curve = pd.read_csv(tuning_path(cfg, method, 'row_norm_thresholds.csv', pair))
    np.testing.assert_allclose(norm_curve.tau, np.arange(1001) / 1000)
    diagnostic = json.loads(tuning_path(cfg, method, 'row_norm_threshold_summary.json', pair).read_text())
    assert diagnostic['parameters'] == {key: selection['parameters'][key] for key in parameters(method)}
    assert diagnostic['diagnostic_only']

    assert not tuning_path(cfg, 'glc', 'selected.json', pair).exists()
    request = {'config': str(config), 'category': method + '_tune', 'condition': pair}
    figures = interactive_figures(request)
    assert len(figures) == (6 if method == 'aga' else 5)
    assert 'Selected α=' in figures[0][1].layout.title.text
    exports = render_preview(request, tmp_path / 'exports')
    assert len(exports) == 2 * len(figures)
    assert all(path.exists() for path in exports)
    for path in exports:
        assert path.read_bytes().startswith(b'%PDF-' if path.suffix == '.pdf' else b'\x89PNG')
    sweep = {'config': str(config), 'category': method + '_tuning_sweeps'}
    curves = interactive_figures(sweep)
    assert len(curves) == 2 * (len(parameters(method)) + 1)
    assert all(len(fig.data[0].y) >= 1 for _, fig in curves)
    assert len(render_preview(sweep, tmp_path / 'sweeps')) == 2 * len(curves)
    run_task(method, cfg, conditions=[pair], show_progress=False)
    assert method in estimators.completed_methods(cfg, [pair])
    assert method in estimators.convergence_methods(cfg, pair)
    frame = pd.read_csv(run_path(cfg, method, 'per_sample.csv', pair))
    assert frame.algorithm.eq('CGAP-SBL-' + method.upper()).all()
    assert {'nmse', 'runtime_s', 'f1'} <= set(frame)
    run_task('comparison', cfg, conditions=[pair], methods=[method], show_progress=False)
    run_task('convergence', cfg, condition=pair, methods=[method], show_progress=False)
    with np.load(task_dir(cfg['run']['output_dir'], 'convergence') / 'convergence.npz') as data:
        assert {method + '_gamma_history', method + '_lambda_history'} <= set(data.files)
    for category in (method + '_run', 'comparison', 'convergence'):
        assert interactive_figures({'config': str(config), 'category': category})


@pytest.mark.parametrize('method', NEW_METHODS)
def test_variant_resume_and_rerun_preserve_other_methods_and_conditions(saved_data, tmp_path, monkeypatch, method):
    cfg = setup(saved_data, method)
    pairs = [(7, 12.), (5, 12.)]
    spec = estimators.ESTIMATORS[method]
    calls = []
    fail = [True]
    def solve(r, C, values, cfg, **kwargs):
        assert set(values) == set(parameters(method))
        assert C is not None
        if fail[0] and len(calls) == 1:
            raise RuntimeError('interrupt variant')
        calls.append(tuple(values.values()))
        return SimpleNamespace(gamma=np.where(r.a, .25, .01), mu=r.Z*.95, iterations=2)
    monkeypatch.setitem(estimators.ESTIMATORS, method, replace(spec, solve=solve))
    with pytest.raises(RuntimeError, match='interrupt variant'):
        run_task(method, cfg, action='tune', conditions=[pairs[0]], show_progress=False)
    fail[0] = False
    run_task(method, cfg, action='tune', conditions=[pairs[0]], show_progress=False, resume=True)
    assert len(calls) == 4
    assert not list(task_dir(cfg['run']['output_dir'], method).glob('tuning/**/.checkpoints/*/*.npz'))
    run_task(method, cfg, action='tune', conditions=[pairs[1]], show_progress=False)
    other = tuning_path(cfg, method, 'selected.json', pairs[1]).read_bytes()
    untouched = tuning_path(cfg, 'glc', 'untouched.txt', pairs[0])
    untouched.parent.mkdir(parents=True)
    untouched.write_text('independent')
    run_task(method, cfg, action='tune', conditions=[pairs[0]], show_progress=False)
    assert len(calls) == 12
    assert tuning_path(cfg, method, 'selected.json', pairs[1]).read_bytes() == other
    assert untouched.read_text() == 'independent'


@pytest.mark.parametrize('method, expected', [('nglc', 'Beta high'), ('rgl', 'Graph coupling κ_g upper bound'),
    ('gd', 'Diffusion time t upper bound'), ('aga', 'Affinity bandwidth h upper bound')])
def test_variant_gui_settings_and_shared_condition_controls(saved_data, tmp_path, monkeypatch, method, expected):
    from streamlit.testing.v1 import AppTest
    cfg = setup(saved_data, method)
    source = tmp_path / 'source.yaml'
    save_yaml(source, cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(source))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    assert not app.exception
    assert len([b for b in app.button if (b.key or '').startswith('open_algorithm_')]) == 9
    app.button(key='open_algorithm_' + method).click().run()
    app.button(key='open_' + method + '_tune').click().run()
    assert not app.exception
    assert expected in [number.label for number in app.number_input]
    assert any(radio.label == 'Condition scope' for radio in app.radio)
    assert any(button.label == 'Add' for button in app.button)
    assert any('TPE: 2 trials × 2' in item.value for item in app.caption)
    if method == 'aga':
        assert 'Gamma floor for logarithms ε_γ' in [number.label for number in app.number_input]


def test_all_spatial_default_budgets_and_canonical_keys():
    from cgap_sbl_jadce.config import load_config
    from cgap_sbl_jadce.settings import validate_spatial_search
    from cgap_sbl_jadce.variants import SPATIAL_METHODS, ALGORITHM_METHODS
    cfg = load_config('configs/experiment.yaml')
    assert set(cfg['algorithms']) == set(ALGORITHM_METHODS)
    expected = {'alpha': (.001, 1000., True), 'beta': (0., 5., False),
                'graph_kappa': (0., 1., False), 'diffusion_time': (.01, 10., True),
                'affinity_bandwidth': (.01, 10., True)}
    for method in SPATIAL_METHODS:
        search = validate_spatial_search(cfg, method)
        assert (search['trials'], search['samples'], search['startup_trials']) == (20 if method == 'aga' else 50, 100, 10)
        assert cfg['algorithms'][method]['tuning']['selection_mode'] == 'pooled'
        assert {key for key in search if key.endswith('_low')} == {name + '_low' for name in parameters(method)}
        for name in parameters(method):
            assert (search[name + '_low'], search[name + '_high'], search[name + '_log']) == expected[name]
