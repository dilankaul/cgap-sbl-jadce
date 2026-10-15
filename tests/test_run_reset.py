from dataclasses import replace
import json
from pathlib import Path

import pytest

from cgap_sbl_jadce.additional_conditions import add_condition
from cgap_sbl_jadce.condition_results import completed_conditions, pending_reset_pairs
from cgap_sbl_jadce.experiments import estimators, pipeline
from cgap_sbl_jadce.gui.workflow import numerical_state
from cgap_sbl_jadce.io import save_json, save_yaml
from cgap_sbl_jadce.paths import run_path, result_path, task_dir, tuning_path
from cgap_sbl_jadce.recovery import begin_recovery, load_recoveries, recovery_path


class Interrupted(RuntimeError):
    pass


def test_all_resets_every_sweep_before_solver_and_preserves_optional_inputs_and_other_methods(saved_data, monkeypatch):
    cfg = saved_data
    estimators.run_estimator_stage(cfg, 'omp', False)
    estimators.run_estimator_stage(cfg, 'cosamp', False, conditions=[(7, 8.)])
    estimators.tune_estimator_stage(cfg, 'sbl', False)
    optional = (9, 10.)
    add_condition(cfg, 'omp', optional)
    pipeline.communication_stage(cfg, False, conditions=[optional], methods=['omp'])
    estimators.run_estimator_stage(cfg, 'omp', False, conditions=[optional])
    protected = [run_path(cfg, 'omp', 'per_sample.csv', optional),
                 run_path(cfg, 'cosamp', 'per_sample.csv', (7, 8.)), tuning_path(cfg, 'sbl', 'selected.json')]
    protected += list(task_dir(cfg['run']['output_dir'], 'communication').rglob('*.npz'))
    protected += list(task_dir(cfg['run']['output_dir'], 'spatial_dependence').rglob('*.npz'))
    note = run_path(cfg, 'omp', 'notes.txt', (7, 8.))
    note.write_text('Keep this note')
    protected.append(note)
    original = {path: path.read_bytes() for path in protected}
    spec = estimators.ESTIMATORS['omp']
    def first_solver(*args, **kwargs):
        for pair in [(7, 8.), (7, 12.), (5, 12.)]:
            assert not run_path(cfg, 'omp', 'per_sample.csv', pair).exists()
        assert {pair for pair, _, _ in completed_conditions(cfg, 'omp')} == {optional}
        summary = json.loads(result_path(cfg['run']['output_dir'], 'omp', 'run_summary.json').read_text())
        assert summary['status'] == 'partial' and summary['completed_pairs'] == [list(optional)]
        assert all(path.read_bytes() == data for path, data in original.items())
        raise Interrupted()
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=first_solver))
    with pytest.raises(Interrupted):
        estimators.run_estimator_stage(cfg, 'omp', False)
    assert all(path.read_bytes() == data for path, data in original.items())


def test_selected_reset_and_resume_use_only_new_checkpoints(saved_data, monkeypatch):
    cfg = saved_data
    estimators.run_estimator_stage(cfg, 'omp', False)
    outside = run_path(cfg, 'omp', 'per_sample.csv', (5, 12.))
    unchanged = outside.read_bytes()
    spec = estimators.ESTIMATORS['omp']
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            assert not run_path(cfg, 'omp', 'per_sample.csv', (7, 12.)).exists()
            raise Interrupted()
        return spec.solve(*args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=fail_second))
    with pytest.raises(Interrupted):
        estimators.run_estimator_stage(cfg, 'omp', False, conditions=[(7, 8.), (7, 12.)])
    assert outside.read_bytes() == unchanged
    calls.clear()
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=lambda *a, **k: (calls.append(1), spec.solve(*a, **k))[1]))
    estimators.run_estimator_stage(cfg, 'omp', False, conditions=[(7, 8.), (7, 12.)], resume=True)
    assert len(calls) == 3
    assert outside.read_bytes() == unchanged
    assert not pending_reset_pairs(cfg, 'omp')


def test_invalid_inputs_preserve_existing_results(saved_data):
    cfg = saved_data
    estimators.run_estimator_stage(cfg, 'omp', False)
    files = list(run_path(cfg, 'omp').rglob('*'))
    original = {path: path.read_bytes() for path in files if path.is_file()}
    path = task_dir(cfg['run']['output_dir'], 'communication') / 'evaluation/0001/observations.npz'
    path.unlink()
    with pytest.raises(FileNotFoundError):
        estimators.run_estimator_stage(cfg, 'omp', False)
    assert all(path.read_bytes() == data for path, data in original.items())


def test_interrupted_reset_resumes_cleanup_before_any_calculation(saved_data, monkeypatch):
    cfg = saved_data
    estimators.run_estimator_stage(cfg, 'omp', False)
    unlink = Path.unlink
    interrupted = run_path(cfg, 'omp', 'per_sample.csv', (5, 12.))
    def fail_cleanup(path, *args, **kwargs):
        if path == interrupted:
            raise Interrupted()
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', fail_cleanup)
    with pytest.raises(Interrupted):
        estimators.run_estimator_stage(cfg, 'omp', False)
    assert pending_reset_pairs(cfg, 'omp') == {(7, 8.), (7, 12.), (5, 12.)}
    assert completed_conditions(cfg, 'omp') == []
    monkeypatch.setattr(Path, 'unlink', unlink)
    spec = estimators.ESTIMATORS['omp']
    calls = []
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=lambda *a, **k: (calls.append(1), spec.solve(*a, **k))[1]))
    estimators.run_estimator_stage(cfg, 'omp', False, resume=True)
    assert len(calls) == 6
    assert numerical_state(cfg, 'omp').status == 'completed'
    assert not pending_reset_pairs(cfg, 'omp')


def test_resume_after_stop_before_reset_still_starts_fresh(saved_data, tmp_path, monkeypatch):
    cfg = saved_data
    estimators.run_estimator_stage(cfg, 'omp', False)
    config = tmp_path / 'config.yaml'
    save_yaml(config, cfg)
    request = {'kind': 'task', 'config': str(config), 'task': 'omp', 'action': 'run'}
    monkeypatch.setenv('CGAP_JOB_DIRECTORY', str(tmp_path / 'original_worker'))
    begin_recovery(request, reset=True)
    monkeypatch.setenv('CGAP_JOB_DIRECTORY', str(tmp_path / 'resumed_worker'))
    save_json(tmp_path / 'resumed_worker/control.json', {'command': 'run'})
    assert completed_conditions(cfg, 'omp') == []
    spec = estimators.ESTIMATORS['omp']
    calls = []
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=lambda *a, **k: (calls.append(1), spec.solve(*a, **k))[1]))
    estimators.run_estimator_stage(cfg, 'omp', False, resume=True)
    assert len(calls) == 6
    assert json.loads(recovery_path(config, 'omp').read_text())['phase'] == 'calculating'


def test_reset_rejects_linked_condition_folder(saved_data, tmp_path):
    cfg = saved_data
    outside = tmp_path / 'outside'
    outside.mkdir()
    sentinel = outside / 'per_sample.csv'
    sentinel.write_text('Keep this data')
    target = run_path(cfg, 'omp', condition=(7, 8.))
    target.parent.mkdir(parents=True)
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='outside the selected'):
        estimators.run_estimator_stage(cfg, 'omp', False)
    assert sentinel.read_text() == 'Keep this data'
    assert not pending_reset_pairs(cfg, 'omp')


def test_overlapping_recovery_is_superseded_and_disjoint_recovery_is_unchanged(saved_data, tmp_path, monkeypatch):
    cfg = saved_data
    config = tmp_path / 'config.yaml'
    save_yaml(config, cfg)
    requests = []
    for pair, name in [((7, 8.), 'old'), ((5, 12.), 'disjoint')]:
        monkeypatch.setenv('CGAP_JOB_DIRECTORY', str(tmp_path / name))
        request = {'kind': 'task', 'config': str(config), 'task': 'omp', 'action': 'run', 'conditions': [list(pair)]}
        begin_recovery(request)
        requests.append(request)
    disjoint = recovery_path(config, 'omp', request=requests[1])
    contents = disjoint.read_bytes()
    old = recovery_path(config, 'omp', request=requests[0])
    monkeypatch.setenv('CGAP_JOB_DIRECTORY', str(tmp_path / 'new'))
    save_json(tmp_path / 'new/control.json', {'command': 'run'})
    estimators.run_estimator_stage(cfg, 'omp', False, conditions=[(7, 8.)])
    assert old.exists() and json.loads(old.read_text())['superseded']
    assert disjoint.read_bytes() == contents
    assert len(load_recoveries(config, 'omp')) == 1


def test_fresh_run_removes_future_condition_checkpoints_before_first_solver(saved_data, monkeypatch):
    cfg = saved_data
    spec = estimators.ESTIMATORS['omp']
    count = []
    def old_failure(*args, **kwargs):
        count.append(1)
        if len(count) == 2:
            raise Interrupted()
        return spec.solve(*args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=old_failure))
    with pytest.raises(Interrupted):
        estimators.run_estimator_stage(cfg, 'omp', False, conditions=[(7, 12.)])
    old = list(run_path(cfg, 'omp', condition=(7, 12.)).glob('.checkpoints/*/*.npz'))
    assert old
    count.clear()
    def fresh_solver(*args, **kwargs):
        if not count:
            assert not any(path.exists() for path in old)
        count.append(1)
        return spec.solve(*args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=fresh_solver))
    estimators.run_estimator_stage(cfg, 'omp', False, conditions=[(7, 8.), (7, 12.)])
    assert len(count) == 4


def test_reset_status_ignores_atomic_temporaries_and_finished_markers(tmp_path, monkeypatch):
    cfg = {'run': {'output_dir': str(tmp_path)}}
    folder = run_path(cfg, 'omp') / '.resets'
    folder.mkdir(parents=True)
    (folder / '.tmp-incomplete.json').write_text('{')
    assert not pending_reset_pairs(cfg, 'omp')
    marker = folder / '0123456789abcdef.json'
    save_json(marker, {'conditions': [[7, 8.]]})
    assert pending_reset_pairs(cfg, 'omp') == {(7, 8.)}
    read = Path.read_text
    def finished(path, *args, **kwargs):
        if path == marker:
            raise FileNotFoundError(path)
        return read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', finished)
    assert not pending_reset_pairs(cfg, 'omp')


@pytest.mark.parametrize('old_scope', [None, [(7, 8.)]])
def test_interrupted_reset_scope_is_preserved_or_fully_replaced(saved_data, monkeypatch, old_scope):
    cfg = saved_data
    estimators.run_estimator_stage(cfg, 'omp', False)
    unlink = Path.unlink
    interrupted = run_path(cfg, 'omp', 'per_sample.csv', (7, 8.))
    def fail(path, *args, **kwargs):
        if path == interrupted:
            raise Interrupted()
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', fail)
    with pytest.raises(Interrupted):
        estimators.run_estimator_stage(cfg, 'omp', False, conditions=old_scope)
    monkeypatch.setattr(Path, 'unlink', unlink)
    if old_scope is None:
        outside = run_path(cfg, 'omp', 'per_sample.csv', (5, 12.))
        original = outside.read_bytes()
        with pytest.raises(ValueError, match='entire condition scope'):
            estimators.run_estimator_stage(cfg, 'omp', False, conditions=[(7, 8.)])
        assert outside.read_bytes() == original
    estimators.run_estimator_stage(cfg, 'omp', False)
    assert not pending_reset_pairs(cfg, 'omp')
    assert numerical_state(cfg, 'omp').status == 'completed'


def test_resume_finishes_interrupted_replacement_of_a_smaller_reset(saved_data, monkeypatch):
    cfg = saved_data
    estimators.run_estimator_stage(cfg, 'omp', False)
    unlink = Path.unlink
    sample = run_path(cfg, 'omp', 'per_sample.csv', (7, 8.))
    def stop_at_sample(path, *args, **kwargs):
        if path == sample:
            raise Interrupted()
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', stop_at_sample)
    with pytest.raises(Interrupted):
        estimators.run_estimator_stage(cfg, 'omp', False, conditions=[(7, 8.)])
    old = next(run_path(cfg, 'omp').glob('.resets/*.json'))
    def stop_at_marker(path, *args, **kwargs):
        if path == old:
            raise Interrupted()
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', stop_at_marker)
    with pytest.raises(Interrupted):
        estimators.run_estimator_stage(cfg, 'omp', False)
    monkeypatch.setattr(Path, 'unlink', unlink)
    estimators.run_estimator_stage(cfg, 'omp', False, resume=True)
    assert not pending_reset_pairs(cfg, 'omp')
    assert numerical_state(cfg, 'omp').status == 'completed'
