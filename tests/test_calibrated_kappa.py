import copy
import json

import pytest

from cgap_sbl_jadce.config import load_config, resolve_activity_kappa
from cgap_sbl_jadce.paths import result_path
from cgap_sbl_jadce.experiments.pipeline import generate_activity_stage


def calibration_config(tmp_path):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path)
    cfg['run']['num_tuning_samples'] = 1
    cfg['run']['num_evaluation_samples'] = 1
    cfg['activity']['kappa_source'] = 'calibration'
    summary = {
        'N': cfg['system']['N'], 'R': cfg['system']['R'],
        'V': cfg['activity']['V'], 'D': cfg['activity']['D'],
        'target_mean_active': cfg['system']['S'],
        'seed': cfg['seeds']['kappa_calibration'],
        'closest_grid_kappa': 2.75,
    }
    from cgap_sbl_jadce.compatibility import preparation_settings
    summary['compatibility'] = preparation_settings(cfg, 'kappa_calibration')
    path = result_path(tmp_path, 'kappa_calibration', 'summary.json')
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(summary))
    return cfg, path, summary


def test_activity_uses_saved_selection_and_preserves_input(tmp_path):
    cfg, path, _ = calibration_config(tmp_path)
    original = copy.deepcopy(cfg)
    generate_activity_stage(cfg, show_progress=False)
    summary = json.loads(result_path(tmp_path, 'activity', 'summary.json').read_text())
    assert summary['kappa'] == 2.75
    assert summary['kappa_source'] == str(path)
    assert load_config(tmp_path / 'config.yaml')['activity']['kappa'] == 2.75
    assert cfg == original


@pytest.mark.parametrize('key', ['N', 'R', 'V', 'D', 'target_mean_active', 'seed'])
def test_incompatible_calibration_is_rejected(tmp_path, key):
    cfg, path, summary = calibration_config(tmp_path)
    summary[key] += 1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match=key):
        resolve_activity_kappa(cfg)


def test_missing_calibration_is_not_silently_replaced(tmp_path):
    cfg, path, _ = calibration_config(tmp_path)
    path.unlink()
    with pytest.raises(FileNotFoundError, match='kappa calibration'):
        resolve_activity_kappa(cfg)
    cfg['activity']['kappa_source'] = 'configured'
    cfg['activity']['kappa'] = 3.0
    assert resolve_activity_kappa(cfg) == (cfg['activity']['kappa'], 'configured')
