from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.settings import pooled_assignment, communication_pairs, grouped_config, analysis_settings, request_settings
from cgap_sbl_jadce.experiments import estimators
from cgap_sbl_jadce.experiments.postprocessing import detection_support
from cgap_sbl_jadce.metrics import precision_recall_f1, support_from_scores
from cgap_sbl_jadce.paths import tuning_path, run_path, task_dir
from cgap_sbl_jadce.variants import TUNED_METHODS

pytestmark = pytest.mark.integration


def pooled_config(saved_data):
    cfg = grouped_config(saved_data)
    cfg['tuning_grid'] = {'pilot_lengths': [7], 'snr_values': [8., 12.]}
    for method in TUNED_METHODS:
        cfg['algorithms'][method]['tuning']['selection_mode'] = 'pooled'
    cfg['analysis'] = {'detection_mode': 'top_k', 'top_k': 3}
    return cfg


def test_default_grid_and_balanced_assignment():
    cfg = load_config('configs/experiment.yaml')
    rows = pooled_assignment(cfg)
    assert [row['q'] for row in rows] == list(range(100))
    counts = Counter((row['L'], row['snr_db']) for row in rows)
    assert len(counts) == 9 and counts[(30, 12.)] == 12
    assert set(counts.values()) == {11, 12}
    assert rows == pooled_assignment(deepcopy(cfg))
    assert len(communication_pairs(cfg, 'tuning')) == 36
    assert communication_pairs(cfg, 'tuning') == communication_pairs(cfg, 'evaluation')
    cfg['tuning_grid']['pilot_length_step'] = 20
    assert len({(r['L'], r['snr_db']) for r in pooled_assignment(cfg)}) == 6
    cfg['run']['num_tuning_samples'] = 5
    with pytest.raises(ValueError, match='at least one'):
        pooled_assignment(cfg)


@pytest.mark.parametrize('method', TUNED_METHODS)
def test_pooled_tuning_uses_one_assigned_solve_per_realization(saved_data, monkeypatch, method):
    cfg = pooled_config(saved_data)
    spec = estimators.ESTIMATORS[method]
    calls = []
    def solve(r, C, parameters, config, **kwargs):
        calls.append((r.Theta.shape[0], r.sigma2))
        assert 'S_q' not in parameters and 'B_q' not in parameters
        return spec.solve(r, C, parameters, config, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, method, replace(spec, solve=solve))
    selected = estimators.tune_estimator_stage(cfg, method, show_progress=False)
    trials = cfg['algorithms'][method]['tuning'].get('trials', 1) if spec.spatial else 1
    # Identical parameter candidates are intentionally cached across trials.
    assert 2 <= len(calls) <= trials * 2
    assert selected['definition']['objective'] == 'mean_top_S_gamma_f1'
    assert selected['calibration']['used_for_parameter_search'] is False
    with np.load(tuning_path(cfg, method, 'final_estimates.npz')) as data:
        assert data['Z_hat'].shape == (2, 20, 3)
        assert data['gamma'].shape == (2, 20)
        expected = np.mean([precision_recall_f1(a, support_from_scores(gamma, int(cfg['system']['S'])))[2]
                            for a, gamma in zip(data['a'], data['gamma'])])
        assert selected['tuning_scores']['f1'] == pytest.approx(expected)
        np.testing.assert_array_equal(data['q'], [0, 1])
    assert len(pd.read_csv(tuning_path(cfg, method, 'assignment.csv'))) == 2
    assert len(pd.read_csv(tuning_path(cfg, method, 'condition_scores.csv'))) == 2
    assert estimators.load_selection(cfg, method, (7, 8.)) == selected
    monkeypatch.setitem(estimators.ESTIMATORS, method, replace(spec, solve=lambda *a, **k: pytest.fail('Repeated completed solve')))
    assert estimators.tune_estimator_stage(cfg, method, show_progress=False, resume=True) == selected


def test_saved_estimates_support_reanalysis_without_solver(saved_data, monkeypatch):
    from cgap_sbl_jadce.condition_results import completed_conditions
    cfg = pooled_config(saved_data)
    estimators.tune_estimator_stage(cfg, 'sbl', show_progress=False)
    pair = (7, 12.)
    estimators.run_estimator_stage(cfg, 'sbl', conditions=[pair], show_progress=False)
    source = run_path(cfg, 'sbl', 'final_estimates.npz', pair)
    saved_bytes = source.read_bytes()
    with np.load(source) as data:
        assert data['Z_hat'].shape == (2, 20, 3)
        assert np.iscomplexobj(data['Z_hat'])
        assert data['gamma'].shape == (2, 20)
    spec = estimators.ESTIMATORS['sbl']
    monkeypatch.setitem(estimators.ESTIMATORS, 'sbl', replace(spec, solve=lambda *a, **k: pytest.fail('Analysis called solver')))
    original_request = request_settings(cfg, {'task': 'sbl'})
    unmasked = None
    for mode, value in [('gamma', 1e100), ('z', 1e100), ('top_k', 0)]:
        cfg['analysis'].update(detection_mode=mode, gamma_threshold=value, z_threshold=value, top_k=0,
                               nmse_mode='support_masked')
        assert request_settings(cfg, {'task': 'sbl'}) == original_request
        assert len(completed_conditions(cfg, 'sbl')) == 1
        estimators.comparison_stage(cfg, conditions=[pair], methods=['sbl'], show_progress=False)
        frame = pd.read_csv(task_dir(cfg['run']['output_dir'], 'comparison') / 'per_sample.csv')
        assert frame.f1.eq(0).all()
        assert frame.nmse.dropna().eq(1).all()
        if unmasked is None:
            unmasked = frame.nmse_unmasked.to_numpy()
        np.testing.assert_allclose(frame.nmse_unmasked, unmasked, equal_nan=True)
        summary = json.loads((task_dir(cfg['run']['output_dir'], 'comparison') / 'summary.json').read_text())
        assert summary['analysis']['sbl']['detection_mode'] == mode
    assert source.read_bytes() == saved_bytes


def test_detector_boundaries_and_deterministic_ties():
    estimate = np.array([[0., 0.], [2., 2.], [1., 1.]], dtype=complex)
    gamma = np.array([0., .5, .5])
    cfg = {'system': {'S': 1, 'N': 3}, 'analysis': {'detection_mode': 'gamma', 'gamma_threshold': .5}}
    np.testing.assert_array_equal(detection_support(estimate, gamma, analysis_settings(cfg)), [False, True, True])
    cfg['analysis'].update(detection_mode='z', z_threshold=1.)
    np.testing.assert_array_equal(detection_support(estimate, gamma, analysis_settings(cfg)), [False, True, True])
    cfg['analysis'].update(detection_mode='top_k', top_k=1)
    np.testing.assert_array_equal(detection_support(estimate, gamma, analysis_settings(cfg)), [False, True, False])
    cfg['analysis']['top_k'] = 0
    assert not detection_support(estimate, gamma, analysis_settings(cfg)).any()
    cfg['analysis']['top_k'] = 4
    with pytest.raises(ValueError, match='0..N'):
        analysis_settings(cfg)


def test_analysis_changes_resume_without_repeating_saved_solver_unit(saved_data, monkeypatch):
    cfg = pooled_config(saved_data)
    pair = (7, 12.)
    spec = estimators.ESTIMATORS['omp']
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('interrupted evaluation')
        return spec.solve(*args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=interrupted))
    with pytest.raises(RuntimeError, match='interrupted evaluation'):
        estimators.run_estimator_stage(cfg, 'omp', conditions=[pair], show_progress=False)
    checkpoint = next(run_path(cfg, 'omp', condition=pair).glob('.checkpoints/*/0000.npz'))
    with np.load(checkpoint) as data:
        original_runtime = json.loads(str(data['_metadata']))['row']['runtime_s']
    cfg['analysis'].update(top_k=0, nmse_mode='support_masked')
    estimators.run_estimator_stage(cfg, 'omp', conditions=[pair], show_progress=False, resume=True)
    assert len(calls) == 3
    frame = pd.read_csv(run_path(cfg, 'omp', 'per_sample.csv', pair)).drop_duplicates('q')
    assert frame.f1.eq(0).all()
    assert frame.loc[frame.q == 0, 'nmse'].iloc[0] == pytest.approx(1.)
    assert frame.loc[frame.q == 0, 'runtime_s'].iloc[0] == pytest.approx(original_runtime)
    assert frame.nmse_mode.eq('support_masked').all()


@pytest.mark.plotting
def test_pooled_interactive_and_native_font_exports(saved_data, tmp_path, monkeypatch):
    from cgap_sbl_jadce.io import save_yaml
    from cgap_sbl_jadce.gui.previews import render_preview, stored_tuning_options
    from cgap_sbl_jadce.gui.charts import interactive_figures
    cfg = pooled_config(saved_data)
    estimators.tune_estimator_stage(cfg, 'gl', show_progress=False)
    config = tmp_path / 'pooled.yaml'
    save_yaml(config, cfg)
    options = stored_tuning_options(cfg, 'gl')
    assert options[0][0] == 'pooled'
    request = {'category': 'gl_tune', 'config': str(config), 'tuning_layout': 'pooled',
               'condition': options[0][1:]}
    spec = estimators.ESTIMATORS['gl']
    monkeypatch.setitem(estimators.ESTIMATORS, 'gl', replace(spec, solve=lambda *a, **k: pytest.fail('Export called solver')))
    figures = dict(interactive_figures(request))
    assert 'Assigned realizations' in figures and 'Mean top-S F1' in figures
    np.testing.assert_allclose(figures['Assigned realizations'].data[0].z, [[1], [1]])
    paths = render_preview(request, tmp_path / 'publication')
    assert {path.suffix for path in paths} == {'.png', '.pdf'}
    assert all(path.exists() and path.stat().st_size > 100 for path in paths)


@pytest.mark.gui
def test_grid_sweeps_and_detection_are_editable_with_explicit_save(saved_data, tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from cgap_sbl_jadce.io import save_yaml
    cfg = pooled_config(saved_data)
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    app_path = Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    app = AppTest.from_file(app_path, default_timeout=30).run()
    next(b for b in app.sidebar.button if b.label == 'Experiment setups').click().run()
    assert not app.exception
    before = config.read_bytes()
    next(w for w in app.number_input if (w.key or '').endswith('_communication_grid_pilot_length_stop')).set_value(9)
    next(w for w in app.number_input if (w.key or '').endswith('_tuning_grid_snr_stop')).set_value(8.)
    next(w for w in app.number_input if (w.key or '').endswith('_evaluation_pilot_length_stop')).set_value(9)
    next(w for w in app.number_input if (w.key or '').endswith('_detector_top_k')).set_value(0)
    next(w for w in app.checkbox if (w.key or '').endswith('_detector_auto_gamma_threshold')).set_value(False)
    next(w for w in app.number_input if (w.key or '').endswith('_detector_gamma_threshold')).set_value(.125)
    assert config.read_bytes() == before
    next(b for b in app.button if b.label == 'Save changes').click().run()
    assert not app.exception and not app.error, [e.value for e in app.error]
    saved = load_config(config)
    assert saved['communication']['grid']['pilot_length_stop'] == 9
    assert saved['tuning_grid']['snr_stop'] == 8.
    assert saved['evaluation']['pilot_length_stop'] == 9
    assert saved['analysis']['top_k'] == 0
    assert saved['analysis']['gamma_threshold'] == .125
    assert all(saved['algorithms'][method]['tuning']['selection_mode'] == 'pooled' for method in TUNED_METHODS)
    assert not list((tmp_path / '.gui').glob('*/job.json'))


def test_pooled_recovery_signature_covers_all_anchor_inputs(saved_data, tmp_path):
    from cgap_sbl_jadce.gui.inputs import input_signature
    from cgap_sbl_jadce.io import save_yaml, save_npz
    from cgap_sbl_jadce.paths import condition_name
    cfg = pooled_config(saved_data)
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    request = {'kind': 'task', 'task': 'gl', 'action': 'tune', 'config': str(config)}
    before = input_signature(request, portable=True)
    # This anchor is different from the method's saved reference condition.
    source = task_dir(cfg['run']['output_dir'], 'communication') / 'tuning' / 'additional' / condition_name((7, 8.)) / '0000' / 'inputs.npz'
    with np.load(source) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['Y'] = arrays['Y'].copy()
    arrays['Y'][0, 0] += .01
    save_npz(source, **arrays)
    assert input_signature(request, portable=True) != before
