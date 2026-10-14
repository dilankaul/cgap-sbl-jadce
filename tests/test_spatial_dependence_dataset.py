import numpy as np
import pytest
from scipy.sparse import issparse
from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.models.activity import ActivitySample
from cgap_sbl_jadce.models.spatial_dependence import build_C
from cgap_sbl_jadce.dataset import save_activity_samples
from cgap_sbl_jadce.io import save_json
from cgap_sbl_jadce.paths import result_path, task_dir
from cgap_sbl_jadce.experiments.pipeline import spatial_dependence_stage
from cgap_sbl_jadce.spatial_dependence_dataset import validate_spatial_dependence_dataset, load_spatial_dependence_matrix


@pytest.fixture
def saved_spatial_dependence(tmp_path):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run'].update(output_dir=str(tmp_path), num_tuning_samples=2, num_evaluation_samples=2)
    cfg['system']['N'] = 20
    cfg['activity'].update(kappa_source='configured', kappa=3.78)
    sample = ActivitySample(np.arange(20) < 3, np.arange(40).reshape(20, 2), np.zeros((2, 2)), np.full(20, .15))
    for split in ['tuning', 'evaluation']:
        save_activity_samples(result_path(tmp_path, 'activity', f'{split}_activity.npz'), [sample, sample])
    from cgap_sbl_jadce.compatibility import preparation_settings
    save_json(result_path(tmp_path, 'activity', 'summary.json'), {'kappa': 3.78, 'D': 15,
        'compatibility': {split: preparation_settings(cfg, 'activity', split) for split in ('tuning', 'evaluation')}})
    spatial_dependence_stage(cfg, show_progress=False)
    return cfg, sample


def test_saved_sparse_matrices_match_formula(saved_spatial_dependence):
    cfg, sample = saved_spatial_dependence
    reference = build_C(sample.device_locations, rho=3.78, U=15)
    for split in ['tuning', 'evaluation']:
        folder = validate_spatial_dependence_dataset(cfg, split)
        for q in [0, 1]:
            C = load_spatial_dependence_matrix(folder, q)
            assert issparse(C) and C.format == 'csr'
            np.testing.assert_array_equal(C.toarray(), reference)
            np.testing.assert_array_equal(C.toarray(), C.T.toarray())
            np.testing.assert_array_equal(C.diagonal(), np.ones(20))
    assert {path.name for path in task_dir(cfg['run']['output_dir'], 'spatial_dependence').iterdir() if path.is_dir()} == {'tuning', 'evaluation'}


def test_incompatible_or_missing_spatial_dependence_rejected(saved_spatial_dependence):
    cfg, _ = saved_spatial_dependence
    summary = result_path(cfg['run']['output_dir'], 'activity', 'summary.json')
    import json
    values = json.loads(summary.read_text())
    values['kappa'] = 4.
    save_json(summary, values)
    with pytest.raises(ValueError, match='rerun spatial dependence'):
        validate_spatial_dependence_dataset(cfg, 'tuning')
    with pytest.raises(FileNotFoundError, match='rerun spatial dependence'):
        load_spatial_dependence_matrix(task_dir(cfg['run']['output_dir'], 'spatial_dependence') / 'evaluation', 100)


def test_selected_spatial_dependence_heatmaps_and_split_progress(saved_spatial_dependence, monkeypatch):
    from cgap_sbl_jadce.figures import spatial_dependence as plotting
    cfg, _ = saved_spatial_dependence
    cfg['figures'].update(tuning_indices=[1], evaluation_indices=[0])
    bars = []
    def progress(indices, **kwargs):
        bars.append(kwargs)
        return indices
    monkeypatch.setattr(plotting, 'tqdm', progress)
    paths = plotting.plot_spatial_dependence(cfg, show_progress=False)
    assert len(paths) == 2 and all(path.exists() for path in paths)
    assert [path.parent.name for path in paths] == ['tuning_0001', 'evaluation_0000']
    assert [bar['desc'] for bar in bars] == ['Spatial dependence figures (tuning)', 'Spatial dependence figures (evaluation)']
    assert all(bar['disable'] for bar in bars)
