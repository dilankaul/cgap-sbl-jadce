from types import SimpleNamespace
import shutil

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.experiments import estimators
from cgap_sbl_jadce.gui.workflow import numerical_state, combined, card_state
from cgap_sbl_jadce.paths import task_dir


def test_fresh_setup_blocks_dependent_tasks_and_algorithm_parents(tmp_path):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path)
    assert numerical_state(cfg, 'kappa_calibration').status == 'ready'
    for task in ('activity', 'communication', 'spatial_dependence', 'omp', 'cosamp', 'comparison', 'convergence'):
        state = numerical_state(cfg, task)
        assert state.status == 'blocked', task
        assert state.detail
    for method in ('sbl', 'anc', 'gl'):
        states = [numerical_state(cfg, method, action) for action in ('tune', 'run')]
        assert all(state.status == 'blocked' for state in states)
        assert combined(states).status == 'blocked'

    assert combined([numerical_state(cfg, task) for task in ('kappa_calibration', 'activity')]).status == 'ready'


def test_badges_unlock_only_when_each_algorithm_own_inputs_are_ready(saved_data):
    shutil.rmtree(task_dir(saved_data['run']['output_dir'], 'spatial_dependence'))
    for method in ('omp', 'cosamp'):
        assert numerical_state(saved_data, method).status == 'ready'
    assert numerical_state(saved_data, 'sbl', 'tune').status == 'ready'
    assert numerical_state(saved_data, 'sbl', 'run').status == 'blocked'
    assert combined([numerical_state(saved_data, 'sbl', action) for action in ('tune', 'run')]).status == 'ready'
    assert combined([numerical_state(saved_data, 'anc', action) for action in ('tune', 'run')]).status == 'blocked'
    assert combined([numerical_state(saved_data, 'gl', action) for action in ('tune', 'run')]).status == 'blocked'
    estimators.tune_estimator_stage(saved_data, 'sbl', show_progress=False)
    assert numerical_state(saved_data, 'sbl', 'tune').status == 'completed'
    assert numerical_state(saved_data, 'sbl', 'run').status == 'ready'
    assert numerical_state(saved_data, 'comparison').status == 'blocked'
    assert numerical_state(saved_data, 'convergence').status == 'blocked'
    estimators.run_estimator_stage(saved_data, 'sbl', show_progress=False)
    assert numerical_state(saved_data, 'sbl', 'run').status == 'completed'
    assert numerical_state(saved_data, 'comparison').status == 'ready'
    assert numerical_state(saved_data, 'convergence').status == 'ready'


def test_active_worker_progress_takes_precedence_over_prerequisite_badge(tmp_path):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path)
    jobs = SimpleNamespace(task_job=lambda *args: {'id':'running-job','status':'running'},
                           log=lambda job: ('', {'label':'Generating data','done':1,'total':10}))
    state = card_state(cfg, tmp_path / 'config.yaml', 'communication', jobs)
    assert state.status == 'running'
    assert state.fraction == .1


def test_saved_comparison_conditions_load_as_lists_and_preserve_source_guards(saved_data, tmp_path):
    import json
    from cgap_sbl_jadce.paths import result_path

    estimators.run_estimator_stage(saved_data, 'omp', show_progress=False)
    estimators.comparison_stage(saved_data, methods=['omp'], show_progress=False)
    path = result_path(saved_data['run']['output_dir'], 'comparison', 'summary.json')
    summary = json.loads(path.read_text())
    assert all(isinstance(condition, list) for condition in summary['conditions'])
    jobs = SimpleNamespace(task_job=lambda *args: None)
    state = card_state(saved_data, tmp_path / 'config.yaml', 'comparison', jobs)
    assert state.status == 'completed'
    assert 'MMV-OMP' in state.detail

    shutil.rmtree(task_dir(saved_data['run']['output_dir'], 'omp'))
    state = numerical_state(saved_data, 'comparison')
    assert state.status == 'incompatible'
    assert 'Source algorithm results' in state.detail
