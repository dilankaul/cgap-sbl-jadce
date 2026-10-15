from dataclasses import replace
import json
import numpy as np
import pytest

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.dataset import save_activity_samples, load_activity_samples
from cgap_sbl_jadce.io import save_json
from cgap_sbl_jadce.paths import result_path, task_dir
from cgap_sbl_jadce.models.activity import ActivitySample
from cgap_sbl_jadce.models.realizations import communication_realization
from cgap_sbl_jadce.communication_dataset import load_communication_sample, validate_communication_dataset
from cgap_sbl_jadce.experiments import pipeline


def test_archives_exactly_match_old_generation_and_share_channels_pilots(saved_data):
    cfg = saved_data
    out = cfg['run']['output_dir']
    for split in ['tuning', 'evaluation']:
        folder = validate_communication_dataset(cfg, split)
        activity = load_activity_samples(result_path(out, 'activity', f'{split}_activity.npz'))
        for q in range(2):
            loaded = load_communication_sample(folder, q)
            for (L, snr), r in loaded.items():
                support = activity[q]
                old = communication_realization(support, cfg['seeds'][f'communication_{split}'], q, 3, L, snr)
                for name in ['a', 'H', 'Z', 'Theta', 'W', 'Y']:
                    np.testing.assert_array_equal(getattr(r, name), getattr(old, name))
                assert r.sigma2 == old.sigma2
                np.testing.assert_allclose(r.Y, r.Theta @ r.Z + r.W)
            with np.load(folder / f'{q:04d}' / 'channels.npz') as data:
                assert data.files.count('H') == 1
    assert len(load_communication_sample(task_dir(out, 'communication') / 'evaluation', 0)) == 3


def test_changed_seed_or_activity_requires_communication_generation(saved_data):
    cfg = saved_data
    cfg['seeds']['communication_tuning'] += 1
    with pytest.raises(ValueError, match='Rerun communication generation'):
        validate_communication_dataset(cfg, 'tuning')
    cfg['seeds']['communication_tuning'] -= 1
    path = result_path(cfg['run']['output_dir'], 'activity', 'tuning_activity.npz')
    samples = load_activity_samples(path)
    samples[0].a[0] = True
    save_activity_samples(path, samples)
    with pytest.raises(ValueError, match='Rerun communication generation'):
        validate_communication_dataset(cfg, 'tuning')


def test_selected_diagnostics_use_saved_samples(saved_data, monkeypatch):
    from cgap_sbl_jadce.figures import communication_diagnostics as plotting
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
    assert [bar['desc'] for bar in bars] == ['Communication figures (tuning)', 'Communication figures (evaluation)']
    assert all(bar['dynamic_ncols'] and not bar['disable'] for bar in bars)
    assert len(paths) == 14
    assert sum('standardized_noise_hsv_heatmap' in path.name for path in paths) == 2
    assert not any('magnitude' in path.name for path in paths)
    assert sum(path.name.startswith('evaluation_') for path in paths) == 7
    assert sum('effective_channel_matrix_hsv_heatmap' in path.name for path in paths) == 2
    assert sum('channel_hsv_heatmap' in path.name for path in paths) == 2
    assert sum('pilot_hsv_heatmap' in path.name for path in paths) == 2
    assert all(path.exists() for path in paths)
    assert sum('effective_channel_row_energy' in path.name for path in paths) == 2
    assert sum('received_signal_hsv_heatmap' in path.name for path in paths) == 2


def test_complex_hsv_encodes_phase_and_magnitude_without_modifying_data():
    from matplotlib.colors import rgb_to_hsv
    from cgap_sbl_jadce.figures.communication_diagnostics import complex_hsv
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
    from cgap_sbl_jadce.figures.communication_diagnostics import plot_communication_preview
    saved_data['figures'].update(tuning_indices=[1], evaluation_indices=[])
    paths = plot_communication_preview(saved_data)
    assert len(paths) == 7
    assert all('tuning_' in path.name and path.stem.endswith('0001') for path in paths)


@pytest.mark.parametrize('indices', [[-1], [2], [True], 'everything'])
def test_invalid_figure_indices_are_rejected(indices):
    from cgap_sbl_jadce.figures.common import selected_realizations
    with pytest.raises(ValueError, match='figures.evaluation_indices'):
        selected_realizations({'figures': {'evaluation_indices': indices}}, 'evaluation', 2)

@pytest.mark.parametrize('split', ['tuning', 'evaluation'])
@pytest.mark.parametrize('count', [0, 3])
def test_all_figure_indices_follow_saved_dataset_count(split, count):
    from cgap_sbl_jadce.figures.common import selected_realizations
    cfg = {'figures': {'tuning_indices': 'all', 'evaluation_indices': 'all'}}
    assert selected_realizations(cfg, split, count) == list(range(count))
