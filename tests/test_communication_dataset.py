from dataclasses import replace
import json
import numpy as np
import pytest

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.dataset import save_activity_samples, load_activity_samples
from cgap_sbl_jadce.io import save_json
from cgap_sbl_jadce.paths import result_path, task_dir
from cgap_sbl_jadce.models.activity import ActivitySample, randomize_activity_sample
from cgap_sbl_jadce.models.realizations import communication_realization
from cgap_sbl_jadce.communication_dataset import load_communication_sample, validate_communication_dataset
from cgap_sbl_jadce.experiments import pipeline


@pytest.fixture
def saved_data(tmp_path):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run'].update(output_dir=str(tmp_path), num_tuning_samples=2, num_evaluation_samples=2)
    cfg['system'].update(N=20, M=3, pilot_lengths=[5, 7], snr_db=[8, 12])
    cfg['algorithms']['K'] = 3
    cfg['tuning'].update(reference_pilot_length=5, alpha_values=[0, 2], beta_values=[0, .1])
    cfg['evaluation'].update(pilot_length=7, gamma_distribution_samples=2)
    a = np.zeros(20, dtype=bool); a[[1, 7, 13]] = True
    sample = ActivitySample(a, np.arange(40).reshape(20, 2), np.zeros((2, 2)), np.full(20, .15))
    evaluation = [sample, replace(sample, a=np.zeros(20, dtype=bool))]
    for name, samples in [('tuning', [sample, sample]), ('evaluation', evaluation),
                          ('randomized_evaluation', [randomize_activity_sample(s, 6031, q) for q, s in enumerate(evaluation)])]:
        save_activity_samples(result_path(tmp_path, 'activity', f'{name}_activity.npz'), samples)
    save_json(result_path(tmp_path, 'activity', 'summary.json'), {'kappa': 3.78, 'D': 15, 'R': 40})
    pipeline.communication_stage(cfg, show_progress=False)
    pipeline.spatial_dependence_stage(cfg, show_progress=False)
    return cfg


def test_archives_exactly_match_old_generation_and_share_channels_pilots(saved_data):
    cfg = saved_data
    out = cfg['run']['output_dir']
    for split in ['tuning', 'evaluation']:
        folder = validate_communication_dataset(cfg, split)
        activity = load_activity_samples(result_path(out, 'activity', f'{split}_activity.npz'))
        randomized = load_activity_samples(result_path(out, 'activity', 'randomized_evaluation_activity.npz'))
        for q in range(2):
            loaded = load_communication_sample(folder, q)
            for (scenario, L, snr), r in loaded.items():
                support = randomized[q] if scenario == 'randomized' else activity[q]
                old = communication_realization(support, cfg['seeds'][f'communication_{split}'], q, 3, L, snr)
                for name in ['a', 'H', 'Z', 'Theta', 'W', 'Y']:
                    np.testing.assert_array_equal(getattr(r, name), getattr(old, name))
                assert r.sigma2 == old.sigma2
                np.testing.assert_allclose(r.Y, r.Theta @ r.Z + r.W)
                if scenario == 'randomized':
                    other = loaded[('correlated', L, snr)]
                    assert r.H is other.H and r.Theta is other.Theta
                    if q == 0:
                        np.testing.assert_allclose(r.W/np.sqrt(r.sigma2), other.W/np.sqrt(other.sigma2))
            with np.load(folder / f'{q:04d}' / 'channels.npz') as data:
                assert data.files.count('H') == 1
    assert len(load_communication_sample(task_dir(out, 3) / 'evaluation', 0)) == 6


def test_changed_seed_or_activity_requires_task03(saved_data):
    cfg = saved_data
    cfg['seeds']['communication_tuning'] += 1
    with pytest.raises(ValueError, match='Rerun Task 03'):
        validate_communication_dataset(cfg, 'tuning')
    cfg['seeds']['communication_tuning'] -= 1
    path = result_path(cfg['run']['output_dir'], 'activity', 'tuning_activity.npz')
    samples = load_activity_samples(path)
    samples[0].a[0] = True
    save_activity_samples(path, samples)
    with pytest.raises(ValueError, match='Rerun Task 03'):
        validate_communication_dataset(cfg, 'tuning')


def test_downstream_tasks_do_not_regenerate_communication(saved_data, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Only Task 03 may generate communication data')
    monkeypatch.setattr(pipeline, 'communication_realization', forbidden)
    monkeypatch.setattr(pipeline, 'paired_activity_realizations', forbidden)
    monkeypatch.setattr(pipeline, 'build_C', forbidden)
    pipeline.tune_stage(saved_data, show_progress=False)
    result = pipeline.evaluate_stage(saved_data, show_progress=False)
    assert result['metric_rows'] == 48
    assert result_path(saved_data['run']['output_dir'], 'evaluation', 'aggregate.csv').exists()
    assert result_path(saved_data['run']['output_dir'], 'evaluation', 'runtime_summary.csv').exists()
    from cgap_sbl_jadce.figures.task06_performance import make_performance_figures
    figures = make_performance_figures(saved_data, show_progress=False)
    assert len(figures) == 10 and all(path.exists() for path in figures)
    assert all('06_estimators' in str(path) for path in figures)
    pipeline.convergence_stage(saved_data)


def test_selected_diagnostics_use_saved_samples(saved_data, monkeypatch):
    from cgap_sbl_jadce.figures import task03_communication_preview as plotting
    original_save = plotting._save
    bars = []
    def capture_progress(indices, **kwargs):
        bars.append(kwargs)
        return indices
    monkeypatch.setattr(plotting, 'tqdm', capture_progress)
    pilot_aspects = []
    def check_save(fig, stem, formats):
        if 'pilot_hsv_heatmap' in stem.name:
            pilot_aspects.append(fig.axes[0].get_aspect())
        return original_save(fig, stem, formats)
    monkeypatch.setattr(plotting, '_save', check_save)
    paths = plotting.plot_communication_preview(saved_data)
    assert pilot_aspects == ['auto', 'auto']
    assert [bar['desc'] for bar in bars] == ['Communication figures (tuning)', 'Communication figures (evaluation)', 'Communication figures (randomized_evaluation)']
    assert all(bar['dynamic_ncols'] and not bar['disable'] for bar in bars)
    assert len(paths) == 17
    assert sum('standardized_noise_hsv_heatmap' in path.name for path in paths) == 2
    assert not any('magnitude' in path.name for path in paths)
    assert sum(path.name.startswith('03_randomized_evaluation_') for path in paths) == 3
    assert sum(path.name.startswith('03_evaluation_') for path in paths) == 7
    assert sum('effective_channel_matrix_hsv_heatmap' in path.name for path in paths) == 3
    assert sum('channel_hsv_heatmap' in path.name for path in paths) == 2
    assert sum('pilot_hsv_heatmap' in path.name for path in paths) == 2
    assert all(path.exists() for path in paths)
    assert sum('effective_channel_row_energy' in path.name for path in paths) == 3
    assert sum('received_signal_hsv_heatmap' in path.name for path in paths) == 3



def test_complex_hsv_encodes_phase_and_magnitude_without_modifying_data():
    from matplotlib.colors import rgb_to_hsv
    from cgap_sbl_jadce.figures.task03_communication_preview import complex_hsv
    matrix = np.array([[0, 1, 1j, -1, -1j, .5]], dtype=complex)
    original = matrix.copy()
    rgb = complex_hsv(matrix, 1.)
    hsv = rgb_to_hsv(rgb)
    np.testing.assert_allclose(hsv[0, 1:, 0], [.5, .75, 0., .25, .5])
    np.testing.assert_allclose(hsv[..., 2], np.abs(matrix))
    np.testing.assert_array_equal(matrix, original)
    np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])
    assert np.isfinite(complex_hsv(np.zeros((2, 2), dtype=complex), 0)).all()


def test_selected_diagnostics_respect_separate_indices(saved_data):
    from cgap_sbl_jadce.figures.task03_communication_preview import plot_communication_preview
    saved_data['figures'].update(tuning_indices=[1], evaluation_indices=[])
    paths = plot_communication_preview(saved_data)
    assert len(paths) == 7
    assert all('03_tuning_' in path.name and path.stem.endswith('0001') for path in paths)


@pytest.mark.parametrize('indices', [[-1], [2], [True], 'everything'])
def test_invalid_figure_indices_are_rejected(indices):
    from cgap_sbl_jadce.figures.common import selected_realizations
    with pytest.raises(ValueError, match='figures.evaluation_indices'):
        selected_realizations({'figures': {'evaluation_indices': indices}}, 'randomized_evaluation', 2)


def test_randomized_figures_share_evaluation_selection():
    from cgap_sbl_jadce.figures.common import selected_realizations
    cfg = {'figures': {'tuning_indices': [1], 'evaluation_indices': [4, 2, 4]}}
    assert selected_realizations(cfg, 'tuning', 5) == [1]
    assert selected_realizations(cfg, 'evaluation', 5) == [4, 2]
    assert selected_realizations(cfg, 'randomized_evaluation', 5) == [4, 2]


def test_combined_tuning_saves_final_thresholds_and_five_figures(saved_data):
    import pandas as pd
    from cgap_sbl_jadce.figures.task05_alpha_beta_tau_tuning import plot_alpha_beta_tau_tuning
    from cgap_sbl_jadce.figures.task05_threshold_scores import plot_threshold_tuning
    selected = pipeline.tune_stage(saved_data, show_progress=False)
    assert selected['definition']['nmse_scope'] == 'true_active_support'
    previous = {**selected, 'definition': {**selected['definition'], 'revision': 2}}
    previous['definition'].pop('nmse_scope')
    with pytest.raises(ValueError, match='Rerun Task 05'):
        pipeline.validate_selection(previous, saved_data)
    out = saved_data['run']['output_dir']
    scores = pd.read_csv(result_path(out, 'tuning', 'alpha_beta_tau_scores.csv'))
    assert len(scores) == 4 * 201
    assert scores.groupby(['alpha', 'beta']).size().eq(201).all()
    for method in ['cgap', 'sbl']:
        curve = pd.read_csv(result_path(out, 'tuning', f'{method}_thresholds.csv'))
        winner = curve.sort_values(['f1', 'recall', 'precision'], ascending=False, kind='stable').iloc[0]
        assert selected[method]['tau'] == pytest.approx(winner.tau)
        assert selected[method]['tuning_f1'] == pytest.approx(winner.f1)
    zero_alpha = scores[scores.alpha.eq(0)]
    np.testing.assert_allclose(zero_alpha[zero_alpha.beta.eq(0)].f1,
                               zero_alpha[zero_alpha.beta.eq(.1)].f1)
    paths = plot_alpha_beta_tau_tuning(saved_data) + plot_threshold_tuning(saved_data)
    assert len(paths) == 5 and all(path.exists() for path in paths)


@pytest.mark.parametrize('split', ['tuning', 'evaluation', 'randomized_evaluation'])
@pytest.mark.parametrize('count', [0, 3])
def test_all_figure_indices_follow_saved_dataset_count(split, count):
    from cgap_sbl_jadce.figures.common import selected_realizations
    cfg = {'figures': {'tuning_indices': 'all', 'evaluation_indices': 'all'}}
    assert selected_realizations(cfg, split, count) == list(range(count))
