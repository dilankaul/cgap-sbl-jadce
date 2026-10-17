from dataclasses import replace
import json
import shutil
import numpy as np
import pandas as pd
import pytest

from cgap_sbl_jadce.experiments import estimators, pipeline
from cgap_sbl_jadce.paths import result_path, task_dir


def test_independent_tasks_use_saved_inputs_and_comparison_runs_no_solvers(saved_data, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('A downstream task regenerated communication or spatial matrices')
    from cgap_sbl_jadce.models import realizations
    monkeypatch.setattr(realizations, 'communication_realization', forbidden)
    monkeypatch.setattr(pipeline, 'build_C', forbidden)
    expected_inputs = {}
    expected_energies = {}
    for method, spec in list(estimators.ESTIMATORS.items()):
        solve = spec.solve
        def capture(r, C, parameters, cfg, _solve=solve, **kwargs):
            key = (r.Y.shape, r.sigma2, r.Y.tobytes())
            expected_inputs[key] = expected_inputs.get(key, 0) + 1
            result = _solve(r, C, parameters, cfg, **kwargs)
            estimate = result.mu if hasattr(result, 'mu') else result
            expected_energies[r.Y.tobytes()] = np.sum(np.abs(estimate) ** 2, axis=1)
            return result
        monkeypatch.setitem(estimators.ESTIMATORS, method, replace(spec, solve=capture))
        if spec.tuning_grid:
            estimators.tune_estimator_stage(saved_data, method, show_progress=False)
        summary = estimators.run_estimator_stage(saved_data, method, show_progress=False)
        assert summary['metric_rows'] == 8
        df = pd.read_csv(result_path(saved_data['run']['output_dir'], method, 'per_sample.csv'))
        assert df.algorithm.eq(spec.name).all()
        assert df.nmse_scope.eq('true_active_support').all()
        assert df[df.q.eq(1)].nmse.isna().all()
        assert df.runtime_s.ge(0).all()
        for path in task_dir(saved_data['run']['output_dir'], method).glob('run/*/selected_channel_energy.npz'):
            with np.load(path) as energy:
                np.testing.assert_array_equal(energy['indices'], [0])
                from cgap_sbl_jadce.communication_dataset import load_communication_sample
                import re
                match = re.fullmatch(r'L(\d+)_snr(.+)dB', path.parent.name)
                pair = (int(match[1]), float(match[2]))
                r = load_communication_sample(task_dir(saved_data['run']['output_dir'], 'communication') / 'evaluation', 0, pair)[pair]
                np.testing.assert_allclose(energy['original'][0], np.sum(np.abs(r.Z) ** 2, axis=1))
                np.testing.assert_allclose(energy['estimated'][0], expected_energies[r.Y.tobytes()])
            summary_path = path.parent / 'summary.json'
            assert 'selected_channel_energy.npz' in json.loads(summary_path.read_text())['outputs']

        if method in ('omp', 'cosamp'):
            assert df.B_q.eq(np.minimum(df.S_q, df.L)).all()
        else:
            assert 'B_q' not in df
            assert result_path(saved_data['run']['output_dir'], method, 'gamma_distribution.npz').exists()

    assert all(count >= len(estimators.ESTIMATORS) for count in expected_inputs.values())
    estimators.convergence_stage(saved_data, show_progress=False)
    for method, spec in list(estimators.ESTIMATORS.items()):
        monkeypatch.setitem(estimators.ESTIMATORS, method, replace(spec, solve=forbidden))
    result = estimators.comparison_stage(saved_data, show_progress=False)
    assert result['metric_rows'] == 8 * len(estimators.ESTIMATORS)
    from cgap_sbl_jadce.figures.algorithm_performance import make_performance_figures
    paths = make_performance_figures(saved_data, show_progress=False)
    assert len(paths) == 6 and all(path.exists() for path in paths)
    from cgap_sbl_jadce.paths import run_path
    raw = run_path(saved_data, 'sbl', 'per_sample.csv', (7, 8))
    old = pd.read_csv(raw)
    old['nmse_scope'] = 'full_matrix'
    old.to_csv(raw, index=False)
    with pytest.raises(ValueError, match='SBL'):
        estimators.comparison_stage(saved_data, show_progress=False, methods=['sbl'])


@pytest.mark.parametrize('method', ['omp', 'cosamp'])
def test_greedy_tasks_require_neither_tuning_nor_spatial_data(saved_data, method, monkeypatch):
    out = saved_data['run']['output_dir']
    shutil.rmtree(task_dir(out, 'spatial_dependence'))
    shutil.rmtree(task_dir(out, 'communication') / 'tuning')
    def forbidden(*args, **kwargs):
        pytest.fail('Greedy task attempted tuning or matrix loading')
    monkeypatch.setattr(estimators, 'tune_estimator', forbidden)
    for other, spec in list(estimators.ESTIMATORS.items()):
        if other != method:
            monkeypatch.setitem(estimators.ESTIMATORS, other, replace(spec, solve=forbidden))
    summary = estimators.run_estimator_stage(saved_data, method, show_progress=False)
    assert summary['status'] == 'complete'
    assert not result_path(out, method, 'tuning/selected.json').exists()


def test_sbl_tunes_independently_without_glc_or_C(saved_data, monkeypatch):
    shutil.rmtree(task_dir(saved_data['run']['output_dir'], 'spatial_dependence'))
    glc = estimators.ESTIMATORS['glc']
    monkeypatch.setitem(estimators.ESTIMATORS, 'glc', replace(glc, solve=lambda *args: pytest.fail('SBL ran GLC')))
    estimators.tune_estimator_stage(saved_data, 'sbl', show_progress=False)
    estimators.run_estimator_stage(saved_data, 'sbl', show_progress=False)
    selected = estimators.load_selection(saved_data, 'sbl')
    assert set(selected['parameters']) == {'tau'}
    assert selected['definition']['method'] == 'SBL'


def test_method_specific_tuning_and_figures(saved_data):
    from cgap_sbl_jadce.figures.glc_parameter_tuning import plot_glc_grid_study
    from cgap_sbl_jadce.experiments.glc_grid_study import glc_grid_study_stage
    from cgap_sbl_jadce.figures.estimator_threshold_scores import plot_threshold_tuning
    from cgap_sbl_jadce.figures.estimator_gamma_distribution import plot_gamma_distribution
    for method in ['sbl', 'glc']:
        estimators.tune_estimator_stage(saved_data, method, show_progress=False)
        estimators.run_estimator_stage(saved_data, method, show_progress=False)
        selected = estimators.load_selection(saved_data, method)
        curve = pd.read_csv(result_path(saved_data['run']['output_dir'], method, 'tuning/thresholds.csv'))
        best = curve.sort_values(['f1', 'recall', 'precision'], ascending=False, kind='stable').iloc[0]
        assert len(curve) == 201
        assert selected['parameters']['tau'] == pytest.approx(best.tau)
        paths = plot_threshold_tuning(saved_data, method) + plot_gamma_distribution(saved_data, method)
        assert len(paths) == 4 and all(path.exists() for path in paths)
    glc_grid_study_stage(saved_data, False)
    scores = pd.read_csv(result_path(saved_data['run']['output_dir'], 'glc_grid_study', 'alpha_beta_tau_scores.csv'))
    assert len(scores) == 4 * 201
    zero = scores[scores.alpha.eq(0)]
    np.testing.assert_allclose(zero[zero.beta.eq(0)].f1, zero[zero.beta.eq(.1)].f1)
    assert len(plot_glc_grid_study(saved_data)) == 3
    path = result_path(saved_data['run']['output_dir'], 'glc', 'tuning/selected.json')
    old = json.loads(path.read_text())
    old['definition'].pop('nmse_scope')
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match='CGAP-SBL-GLC'):
        estimators.load_selection(saved_data, 'glc')


def test_runtime_excludes_metrics_and_archive_loading(saved_data, monkeypatch):
    elapsed = {'t': 0.}
    monkeypatch.setattr(estimators.time, 'perf_counter', lambda: elapsed['t'])
    spec = estimators.ESTIMATORS['omp']
    def timed(*args, **kwargs):
        result = spec.solve(*args, **kwargs)
        elapsed['t'] += .01
        return result
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=timed))
    for name in ['nmse', 'precision_recall_f1', 'load_communication_sample']:
        operation = getattr(estimators, name)
        def untimed(*args, _operation=operation, **kwargs):
            elapsed['t'] += 100.
            return _operation(*args, **kwargs)
        monkeypatch.setattr(estimators, name, untimed)
    estimators.run_estimator_stage(saved_data, 'omp', show_progress=False)
    df = pd.read_csv(result_path(saved_data['run']['output_dir'], 'omp', 'per_sample.csv'))
    np.testing.assert_allclose(df.runtime_s, .01)


def test_comparison_reports_missing_algorithm(saved_data):
    with pytest.raises(ValueError, match='Select at least one'):
        estimators.comparison_stage(saved_data, show_progress=False)


@pytest.mark.parametrize('method', ['sbl', 'glc'])
def test_tune_uses_only_tuning_inputs_and_run_reuses_selection(saved_data, method, monkeypatch):
    out = saved_data['run']['output_dir']
    folder = task_dir(out, 'communication') / 'evaluation'
    held_out = folder.with_name('held_out')
    folder.rename(held_out)
    selected = estimators.tune_estimator_stage(saved_data, method, show_progress=False)
    assert not result_path(out, method, 'per_sample.csv').exists()
    assert not result_path(out, method, 'run_summary.json').exists()
    assert selected['definition']['method'] == estimators.ESTIMATORS[method].name
    held_out.rename(folder)
    selection_path = result_path(out, method, 'tuning/selected.json')
    original = selection_path.read_bytes()
    monkeypatch.setattr(estimators, 'tune_estimator', lambda *args, **kwargs: pytest.fail('Run retuned the algorithm'))
    estimators.run_estimator_stage(saved_data, method, show_progress=False)
    assert selection_path.read_bytes() == original
    assert result_path(out, method, 'per_sample.csv').exists()


@pytest.mark.parametrize('method', ['sbl', 'glc'])
def test_run_requires_tuning_without_executing_a_solver(saved_data, method, monkeypatch):
    spec = estimators.ESTIMATORS[method]
    monkeypatch.setitem(estimators.ESTIMATORS, method, replace(spec, solve=lambda *args: pytest.fail('Ran before tuning')))
    with pytest.raises(FileNotFoundError, match='Tune'):
        estimators.run_estimator_stage(saved_data, method, show_progress=False)
    assert not result_path(saved_data['run']['output_dir'], method, 'run_summary.json').exists()
