import copy
import json
from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.io import save_json
from cgap_sbl_jadce.paths import result_path, task_dir
from cgap_sbl_jadce.provenance import save_task_snapshot


def test_task_snapshot_uses_activity_data_not_current_calibration(tmp_path):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path)
    original = copy.deepcopy(cfg)
    save_json(result_path(tmp_path, 'activity', 'summary.json'), {'kappa': 2.75, 'D': 15., 'N': 200})
    save_json(result_path(tmp_path, 'kappa_calibration', 'summary.json'), {'closest_grid_kappa': 3.5})
    save_task_snapshot(cfg, 'cosamp', workers=2, selected_parameters={'sbl': {'tau': 0.02}})
    saved = json.loads((task_dir(tmp_path, 'cosamp') / 'manifest.json').read_text())['configuration']
    assert saved['activity']['kappa'] == 2.75
    assert saved['task_runtime']['selected_parameters']['sbl']['tau'] == 0.02
    assert saved['task_runtime']['workers'] == 2
    assert json.loads((task_dir(tmp_path, 'cosamp') / 'manifest.json').read_text())['task'] == 'cosamp'
    assert (tmp_path / 'config.yaml').exists()
    assert not (task_dir(tmp_path, 'cosamp') / 'config.yaml').exists()
    assert cfg == original


def test_snapshot_preserves_communication_manifest_fields(tmp_path):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path)
    manifest_path = task_dir(tmp_path, 'communication') / 'manifest.json'
    save_json(manifest_path, {'communication_seed': 123, 'M': 4, 'Theta_shape': [20, 200]})
    save_task_snapshot(cfg, 'communication')
    saved = json.loads(manifest_path.read_text())
    assert saved['communication_seed'] == 123
    assert saved['Theta_shape'] == [20, 200]
    assert saved['task'] == 'communication'


def test_later_tasks_preserve_root_yaml_and_record_exact_config(tmp_path):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path)
    cfg['activity']['kappa'] = 2.75
    save_task_snapshot(cfg, 'activity')
    root = tmp_path / 'config.yaml'
    original = root.read_bytes()
    save_json(result_path(tmp_path, 'activity', 'summary.json'), {'kappa': 2.75, 'D': 15.})
    later = copy.deepcopy(cfg)
    later['algorithms']['K'] += 1
    save_task_snapshot(later, 'omp', workers=3)
    assert root.read_bytes() == original
    manifest = json.loads((task_dir(tmp_path, 'omp') / 'manifest.json').read_text())
    assert manifest['configuration']['algorithms']['K'] == later['algorithms']['K']
    assert len(list(tmp_path.rglob('*.yaml'))) == 1
