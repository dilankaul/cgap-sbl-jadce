import numpy as np
import pytest

from cgap_sbl_jadce.config import load_config, activity_correlation_parameters
from cgap_sbl_jadce.experiments.pipeline import evaluation_conditions, validate_selection, _evaluate_sample
from cgap_sbl_jadce.models.activity import ActivitySample, randomize_activity_sample
from cgap_sbl_jadce.models.realizations import paired_activity_realizations
from cgap_sbl_jadce.io import save_json
from cgap_sbl_jadce.paths import result_path


def sample():
    a = np.zeros(20, dtype=bool)
    a[[1, 7, 13]] = True
    return ActivitySample(a, np.arange(40).reshape(20, 2), np.zeros((2, 2)), np.zeros(20))


def test_paired_support_geometry_channels_pilots_and_standardized_noise():
    activity = sample()
    original = activity.a.copy()
    randomized_sample = randomize_activity_sample(activity, 6031, 7)
    pairs = paired_activity_realizations(activity, randomized_sample, 4283, 7, 3, 10, 12)
    correlated, randomized = pairs.values()
    assert np.array_equal(activity.a, original)
    assert correlated.a.sum() == randomized.a.sum() == 3
    assert not np.array_equal(correlated.a, randomized.a)
    np.testing.assert_array_equal(correlated.H, randomized.H)
    np.testing.assert_array_equal(correlated.Theta, randomized.Theta)
    np.testing.assert_allclose(correlated.W / np.sqrt(correlated.sigma2), randomized.W / np.sqrt(randomized.sigma2))
    for r in pairs.values():
        np.testing.assert_allclose(r.Z, r.a[:, None] * r.H.T)
        assert np.isclose(np.mean(abs(r.Theta @ r.Z)**2) / r.sigma2, 10**1.2)
    other = paired_activity_realizations(activity, randomized_sample, 4283, 7, 3, 15, 24)
    np.testing.assert_array_equal(randomized.a, other['randomized'].a)
    repeated = paired_activity_realizations(activity, randomized_sample, 4283, 7, 3, 10, 12)
    np.testing.assert_array_equal(randomized.Y, repeated['randomized'].Y)


def test_paper_conditions_and_saved_correlation_parameters(tmp_path):
    cfg = load_config('configs/experiment.yaml')
    assert all(L == 30 for L, _, sweep in evaluation_conditions(cfg) if sweep == 'snr_sweep')
    assert all(snr == 12 for _, snr, sweep in evaluation_conditions(cfg) if sweep == 'pilot_sweep')
    cfg['run']['output_dir'] = str(tmp_path)
    save_json(result_path(tmp_path, 'activity', 'summary.json'), {'kappa': 3.78, 'D': 15})
    cfg['activity']['D'] = 99
    assert activity_correlation_parameters(cfg) == {'rho': 3.78, 'U': 15}
    with pytest.raises(ValueError, match='Rerun Tasks 05 and 06'):
        validate_selection({'definition': {'method': 'legacy'}, 'alpha': 1}, cfg)


def test_evaluation_runs_six_methods_scenarios_without_oracle_information_in_sbl(tmp_path):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    selected = {'cgap': {'alpha': 2, 'beta': .1, 'tau': .05}, 'sbl': {'tau': .05},
                'reference_pilot_length': 10, 'reference_snr_db': 12}
    alg = {**cfg['algorithms'], 'K': 3}
    from cgap_sbl_jadce.communication_dataset import save_communication_sample
    paired = paired_activity_realizations(sample(), randomize_activity_sample(sample(), 6031, 0), 4283, 0, 3, 10, 12)
    save_communication_sample(tmp_path, 0, {(scenario, 10, 12): r for scenario, r in paired.items()})
    rows, gamma = _evaluate_sample((0, sample(), tmp_path, {'N': 20, 'M': 3},
                                   {'rho': 3.78, 'U': 15}, alg, selected, [(10, 12, 'pilot_sweep')], True))
    assert len(rows) == 6
    assert {row['S_q'] for row in rows} == {3}
    assert len([row for row in rows if row['scenario'] == 'randomized']) == 2
    for row in rows:
        assert ('B_q' in row) == row['algorithm'].startswith('MMV-')
    assert set(gamma) == {'a', 'cgap_gamma', 'sbl_gamma'}


def test_aggregation_preserves_scenarios_and_averages_paired_advantages(tmp_path):
    import pandas as pd
    from cgap_sbl_jadce.experiments.pipeline import aggregate_stage
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path)
    save_json(result_path(tmp_path, 'activity', 'summary.json'), {'kappa': 3.78, 'D': 15})
    rows = []
    for scenario, gain in [('correlated', .2), ('randomized', -.1)]:
        for q in range(2):
            for method in ['CGAP-SBL-ANC', 'SBL']:
                rows.append(dict(q=q, scenario=scenario, algorithm=method, L=20, snr_db=12,
                                 sweep='snr_sweep', precision=.5, recall=.5, runtime_s=.1,
                                 f1=.5 + (gain if method == 'CGAP-SBL-ANC' else 0),
                                 nmse=(np.nan if q == 0 else (.2 if method == 'CGAP-SBL-ANC' else .3))))
    path = result_path(tmp_path, 'evaluation', 'per_sample.csv')
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    aggregate_stage(cfg)
    aggregate = pd.read_csv(result_path(tmp_path, 'evaluation', 'aggregate.csv'))
    assert len(aggregate) == 4
    assert not aggregate.nmse.isna().any()
    advantages = pd.read_csv(result_path(tmp_path, 'evaluation', 'spatial_randomization_advantages.csv')).set_index('scenario')
    assert np.isclose(advantages.loc['correlated', 'f1_gain'], .2)
    assert np.isclose(advantages.loc['randomized', 'f1_gain'], -.1)
    np.testing.assert_allclose(advantages.nmse_reduction, .1)


def test_task02_saves_randomized_pairs_with_original_seed_sequence(tmp_path):
    from cgap_sbl_jadce.dataset import load_activity_samples
    from cgap_sbl_jadce.experiments.pipeline import generate_activity_stage
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run'].update(output_dir=str(tmp_path), num_tuning_samples=2, num_evaluation_samples=3)
    cfg['activity'].update(kappa_source='configured', kappa=3.78)
    summary = generate_activity_stage(cfg, show_progress=False)
    correlated = load_activity_samples(result_path(tmp_path, 'activity', 'evaluation_activity.npz'))
    randomized = load_activity_samples(result_path(tmp_path, 'activity', 'randomized_evaluation_activity.npz'))
    assert len(randomized) == len(correlated) == 3
    for q, (original, saved) in enumerate(zip(correlated, randomized)):
        rng = np.random.default_rng(np.random.SeedSequence([6031, q]))
        expected = np.zeros_like(original.a)
        expected[rng.choice(len(expected), size=int(original.a.sum()), replace=False)] = True
        np.testing.assert_array_equal(saved.a, expected)
        np.testing.assert_array_equal(saved.device_locations, original.device_locations)
        np.testing.assert_array_equal(saved.event_locations, original.event_locations)
        np.testing.assert_allclose(saved.activation_probabilities, original.a.sum()/len(original.a))
    assert summary['randomized_evaluation']['seed'] == 6031
    assert summary['randomized_evaluation']['mean_active'] == summary['evaluation']['mean_active']


def test_randomized_spatial_figures_omit_event_model_overlays(tmp_path, monkeypatch):
    from cgap_sbl_jadce.figures import task02_activity_realizations as plotting
    captured = {}
    monkeypatch.setattr(plotting, 'plot_activity_sample', lambda sample, **kwargs: captured.update(kwargs) or [])
    cfg = {'run': {'output_dir': str(tmp_path)}}
    plotting._plot_activity_realization_sample(cfg, sample(), 'randomized_evaluation', 7, 'png',
        {'show_probability_field': True, 'show_event_radius': True})
    assert captured['show_probability_field'] is False
    assert captured['show_event_radius'] is False
    assert captured['show_events'] is False
    assert captured['stem'].parent.name == 'evaluation_0007'
    assert '3 active MTDs' in captured['title']
    assert '7' in captured['title']


def test_communication_rejects_mismatched_saved_pairs():
    from dataclasses import replace
    original = sample()
    randomized = randomize_activity_sample(original, 6031, 0)
    with pytest.raises(ValueError, match='matching counts and geometry'):
        paired_activity_realizations(original, replace(randomized, a=np.zeros(20, dtype=bool)), 4283, 0, 3, 10, 12)
