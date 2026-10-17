from copy import deepcopy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.settings import validate_spatial_search, grouped_config, request_settings
from cgap_sbl_jadce.experiments.spatial_tuning import parameter_distributions
from cgap_sbl_jadce.variants import SPATIAL_METHODS, parameters


@pytest.mark.parametrize('method', SPATIAL_METHODS)
def test_default_domains_and_optuna_scales(method):
    cfg = load_config('configs/experiment.yaml')
    settings = validate_spatial_search(cfg, method)
    distributions = parameter_distributions(settings, method)
    expected = {'alpha': (.001, 1000., True), 'beta': (0., 5., False),
                'graph_kappa': (0., float(np.nextafter(1., 0.)), False),
                'diffusion_time': (.01, 10., True), 'affinity_bandwidth': (.01, 10., True)}
    assert set(distributions) == set(parameters(method))
    for name, dist in distributions.items():
        assert (dist.low, dist.high, dist.log) == expected[name]


@pytest.mark.parametrize('method, parameter', [('glc', 'alpha'), ('nglc', 'beta'),
    ('rgl', 'graph_kappa'), ('gd', 'diffusion_time'), ('aga', 'affinity_bandwidth')])
def test_logarithmic_sampling_rejects_zero_lower_bound(method, parameter):
    cfg = load_config('configs/experiment.yaml')
    cfg['algorithms'][method]['tuning'].update({parameter + '_low': 0., parameter + '_log': True})
    with pytest.raises(ValueError, match='strictly positive'):
        validate_spatial_search(cfg, method)
    cfg['algorithms'][method]['tuning'][parameter + '_log'] = False
    if parameter == 'affinity_bandwidth':
        with pytest.raises(ValueError, match='strictly positive'):
            validate_spatial_search(cfg, method)
    else:
        assert validate_spatial_search(cfg, method)[parameter + '_log'] is False
    cfg['algorithms'][method]['tuning'][parameter + '_log'] = 'true'
    with pytest.raises(ValueError, match='true or false'):
        validate_spatial_search(cfg, method)


def test_rgl_upper_setting_one_samples_below_gl_boundary():
    import optuna
    from cgap_sbl_jadce.algorithms.graph_interactions import spatial_penalty
    cfg = load_config('configs/experiment.yaml')
    distributions = parameter_distributions(validate_spatial_search(cfg, 'rgl'), 'rgl')
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=17389))
    study.enqueue_trial({'alpha': .001, 'graph_kappa': distributions['graph_kappa'].high})
    trial = study.ask(fixed_distributions=distributions)
    assert 0 <= trial.params['graph_kappa'] < 1
    spatial_penalty(np.eye(3), 3, 'rgl', **trial.params)
    study.tell(trial, 0.)
    for _ in range(12):
        trial = study.ask(fixed_distributions=distributions)
        assert 0 <= trial.params['graph_kappa'] < 1
        study.tell(trial, trial.params['graph_kappa'])


def test_scales_belong_to_each_algorithm_independently():
    cfg = load_config('configs/experiment.yaml')
    before = deepcopy(cfg)
    cfg['algorithms']['gl']['tuning']['alpha_log'] = False
    assert request_settings(cfg, {'task': 'gl', 'action': 'tune'}) != request_settings(before, {'task': 'gl', 'action': 'tune'})
    for method in set(SPATIAL_METHODS) - {'gl'}:
        assert request_settings(cfg, {'task': method, 'action': 'tune'}) == request_settings(before, {'task': method, 'action': 'tune'})
        assert parameter_distributions(validate_spatial_search(cfg, method), method)['alpha'].log is True


@pytest.mark.integration
def test_log_tpe_records_actual_distributions_and_scores_all_samples(saved_data, monkeypatch):
    from cgap_sbl_jadce.experiments import estimators
    from cgap_sbl_jadce.paths import tuning_path
    cfg = grouped_config(saved_data)
    cfg['tuning_grid'] = {'pilot_lengths': [7], 'snr_values': [8., 12.]}
    cfg['algorithms']['gl']['tuning'].update(selection_mode='pooled', alpha_low=.001, alpha_high=1000.,
                                             alpha_log=True, trials=3, startup_trials=1)
    spec = estimators.ESTIMATORS['gl']
    calls = []
    def solve(r, C, parameters, config, **kwargs):
        calls.append(parameters['alpha'])
        return SimpleNamespace(gamma=np.where(r.a, .2, .01), mu=r.Z.copy(), iterations=2)
    monkeypatch.setitem(estimators.ESTIMATORS, 'gl', replace(spec, solve=solve))
    selected = estimators.tune_estimator_stage(cfg, 'gl', show_progress=False)
    assert len(calls) == 6 and min(calls) >= .001 and max(calls) <= 1000.
    space = json.loads(tuning_path(cfg, 'gl', 'search_space.json').read_text())
    assert space == {'alpha': {'low': .001, 'high': 1000., 'log': True}}
    assert selected['definition']['parameters']['alpha_log'] is True
    summary = json.loads(tuning_path(cfg, 'gl', 'summary.json').read_text())
    assert 'search_space.json' in summary['outputs']
    diagnostic = json.loads(tuning_path(cfg, 'gl', 'row_norm_threshold_summary.json').read_text())
    assert diagnostic['diagnostic_only'] is False and diagnostic['used_for_parameter_search'] is False


@pytest.mark.gui
def test_each_algorithm_has_independent_sampling_controls(saved_data, tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from cgap_sbl_jadce.io import save_yaml
    cfg = grouped_config(saved_data)
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    app_path = Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    app = AppTest.from_file(app_path, default_timeout=30).run()
    next(b for b in app.sidebar.button if b.label == 'Experiment setups').click().run()
    assert not app.exception
    for method in SPATIAL_METHODS:
        for name in parameters(method):
            assert len([w for w in app.checkbox if (w.key or '').endswith('_' + method + '_search_' + name + '_log')]) == 1
    next(w for w in app.number_input if (w.key or '').endswith('_gl_search_alpha_low')).set_value(.001)
    next(w for w in app.checkbox if (w.key or '').endswith('_gl_search_alpha_log')).set_value(True)
    next(b for b in app.button if b.label == 'Save changes').click().run()
    assert not app.exception and not app.error
    result = load_config(config)
    assert result['algorithms']['gl']['tuning']['alpha_log'] is True
    assert result['algorithms']['gl']['tuning']['alpha_low'] == .001
    assert result['algorithms']['glc']['tuning']['alpha_log'] is False
    assert result['algorithms']['glc']['tuning']['alpha_low'] == 0.
