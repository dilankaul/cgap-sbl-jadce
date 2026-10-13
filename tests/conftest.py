from dataclasses import replace
import numpy as np
import pytest
from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.dataset import save_activity_samples
from cgap_sbl_jadce.models.activity import ActivitySample
from cgap_sbl_jadce.io import save_json
from cgap_sbl_jadce.paths import result_path
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
    for name, samples in [('tuning', [sample, sample]), ('evaluation', evaluation)]:
        save_activity_samples(result_path(tmp_path, 'activity', f'{name}_activity.npz'), samples)
    save_json(result_path(tmp_path, 'activity', 'summary.json'), {'kappa': 3.78, 'D': 15, 'R': 40})
    pipeline.communication_stage(cfg, show_progress=False)
    pipeline.spatial_dependence_stage(cfg, show_progress=False)
    return cfg
