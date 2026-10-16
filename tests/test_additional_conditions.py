from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest

from cgap_sbl_jadce.additional_conditions import add_condition, additional_conditions, input_reasons
from cgap_sbl_jadce.communication_dataset import load_communication_sample, validate_communication_dataset
from cgap_sbl_jadce.condition_results import completed_conditions
from cgap_sbl_jadce.experiments import estimators, pipeline
from cgap_sbl_jadce.experiments.readiness import blocking_reasons
from cgap_sbl_jadce.gui.algorithm_actions import completion_plan
from cgap_sbl_jadce.gui.workflow import numerical_state
from cgap_sbl_jadce.io import save_yaml
from cgap_sbl_jadce.paths import run_path, task_dir
from cgap_sbl_jadce.settings import grouped_config

PAIR = (9, 10.)


def test_optional_registration_is_separate_and_shared(saved_data, tmp_path):
    config = tmp_path / 'source.yaml'
    save_yaml(config, saved_data)
    original = config.read_bytes()
    base = estimators.evaluation_conditions(saved_data)
    add_condition(saved_data, 'omp', PAIR)
    add_condition(saved_data, 'omp', PAIR)
    assert additional_conditions(saved_data, 'omp') == [PAIR]
    assert additional_conditions(saved_data, 'cosamp') == [PAIR]
    assert estimators.evaluation_conditions(saved_data) == base
    assert estimators.evaluation_conditions(saved_data, [PAIR], method='omp') == [(*PAIR, 'condition')]
    assert estimators.evaluation_conditions(saved_data, [PAIR], method='cosamp') == [(*PAIR, 'condition')]
    assert config.read_bytes() == original
    assert input_reasons(saved_data, 'omp', PAIR)
    add_condition(saved_data, 'omp', (7, 8))
    assert additional_conditions(saved_data, 'omp') == [PAIR]


def test_preparation_preserves_base_inputs_and_results(saved_data):
    estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[(7, 8)])
    folder = task_dir(saved_data['run']['output_dir'], 'communication')
    original = {path: path.read_bytes() for path in folder.glob('*/*/*.npz')}
    add_condition(saved_data, 'omp', PAIR)
    pipeline.communication_stage(saved_data, False, conditions=[PAIR], methods=['omp'])
    assert not input_reasons(saved_data, 'omp', PAIR)
    assert all(path.read_bytes() == data for path, data in original.items())
    assert estimators.completed_methods(saved_data, [(7, 8)]) == ['omp']
    r = load_communication_sample(folder / 'evaluation', 0, PAIR)[PAIR]
    base = load_communication_sample(folder / 'evaluation', 0, (7, 8))[(7, 8)]
    for name in ('H', 'a', 'Z'):
        np.testing.assert_array_equal(getattr(r, name), getattr(base, name))
    np.testing.assert_allclose(r.Y, r.Theta @ r.Z + r.W)
    assert r.Theta.shape == (9, 20)
    assert (folder / 'tuning/additional').exists()
    extra = {path: path.read_bytes() for path in folder.glob('*/additional/*/*/*.npz')}
    pipeline.communication_stage(saved_data, False, conditions=[PAIR], methods=['omp'], resume=True)
    assert all(path.read_bytes() == data for path, data in extra.items())


def test_optional_run_is_explicit_and_does_not_affect_sweep_completion(saved_data, tmp_path, monkeypatch):
    add_condition(saved_data, 'omp', PAIR)
    pipeline.communication_stage(saved_data, False, conditions=[PAIR], methods=['omp'])
    spec = estimators.ESTIMATORS['omp']
    lengths = []
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec,
        solve=lambda r, *a, **k: (lengths.append(r.Theta.shape[0]), spec.solve(r, *a, **k))[1]))
    summary = estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[PAIR])
    assert lengths == [9, 9] and summary['status'] == 'partial'
    assert summary['total_conditions'] == 3
    assert numerical_state(saved_data, 'omp').status == 'ready'
    config = tmp_path / 'source.yaml'; save_yaml(config, saved_data)
    assert completion_plan(config, 'omp', [PAIR]).completed == [PAIR]
    assert PAIR not in completion_plan(config, 'omp').pairs
    lengths.clear()
    estimators.run_estimator_stage(saved_data, 'omp', False, resume=True)
    assert lengths == [7, 7, 7, 7, 5, 5]
    assert numerical_state(saved_data, 'omp').status == 'completed'
    assert PAIR in {pair for pair, _, _ in completed_conditions(saved_data, 'omp')}
    assert estimators.comparison_stage(saved_data, False, conditions=[PAIR], methods=['omp'])['metric_rows'] == 2


@pytest.mark.parametrize('policy', ['reference', 'per_condition'])
def test_optional_sbl_respects_independent_tuning_policy(saved_data, policy):
    cfg = grouped_config(saved_data)
    cfg['algorithms']['sbl']['tuning']['selection_mode'] = policy
    if policy == 'per_condition':
        pipeline.communication_stage(cfg, False)
    estimators.tune_estimator_stage(cfg, 'sbl', False, conditions=[(5, 12)])
    add_condition(cfg, 'sbl', PAIR)
    pipeline.communication_stage(cfg, False, conditions=[PAIR], methods=['sbl'])
    folder = task_dir(cfg['run']['output_dir'], 'communication')
    if policy == 'per_condition':
        assert (folder / 'tuning/additional').exists()
        assert blocking_reasons(cfg, 'sbl', conditions=[PAIR])
        assert PAIR not in estimators.tuning_conditions(cfg, method='sbl')
        estimators.tune_estimator_stage(cfg, 'sbl', False, conditions=[PAIR])
    else:
        assert (folder / 'tuning/additional').exists()
    assert not blocking_reasons(cfg, 'sbl', conditions=[PAIR])
    estimators.run_estimator_stage(cfg, 'sbl', False, conditions=[PAIR])
    assert estimators.convergence_methods(cfg, PAIR) == ['sbl']
    summary = estimators.convergence_stage(cfg, False, methods=['sbl'], condition=PAIR)
    assert summary['status'] == 'complete'


def test_optional_input_failure_resumes_without_repeating_finished_unit(saved_data, monkeypatch):
    from cgap_sbl_jadce.models import realizations
    add_condition(saved_data, 'omp', PAIR)
    generate = realizations.communication_realization
    calls = []
    def fail_second(*args):
        calls.append(args[2])
        if len(calls) == 2:
            raise RuntimeError('interrupted')
        return generate(*args)
    monkeypatch.setattr(realizations, 'communication_realization', fail_second)
    with pytest.raises(RuntimeError, match='interrupted'):
        pipeline.communication_stage(saved_data, False, conditions=[PAIR], methods=['omp'])
    assert input_reasons(saved_data, 'omp', PAIR)
    calls.clear()
    monkeypatch.setattr(realizations, 'communication_realization', lambda *a: (calls.append(a[2]), generate(*a))[1])
    pipeline.communication_stage(saved_data, False, conditions=[PAIR], methods=['omp'], resume=True)
    assert calls == [1, 0, 1]
    assert not input_reasons(saved_data, 'omp', PAIR)
    assert not list(task_dir(saved_data['run']['output_dir'], 'communication').rglob('.checkpoints/*/*.npz'))


def test_shared_deletion_removes_options_but_preserves_inputs_and_results(saved_data):
    from cgap_sbl_jadce.additional_conditions import remove_condition
    add_condition(saved_data, 'omp', PAIR)
    pipeline.communication_stage(saved_data, False, conditions=[PAIR], methods=['omp'])
    estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[PAIR])
    paths = [run_path(saved_data, 'omp', 'per_sample.csv', PAIR),
             task_dir(saved_data['run']['output_dir'], 'communication') / 'evaluation/additional/L0009_snr10dB/0000/inputs.npz']
    before = {path: path.read_bytes() for path in paths}
    remove_condition(saved_data, PAIR)
    for method in estimators.ESTIMATORS:
        assert additional_conditions(saved_data, method) == []
        assert PAIR not in {value[:2] for value in estimators.evaluation_conditions(saved_data, method=method, include_optional=True)}
    assert all(path.read_bytes() == data for path, data in before.items())
    add_condition(saved_data, 'sbl', PAIR)
    assert estimators.completed_methods(saved_data, [PAIR]) == ['omp']


@pytest.mark.parametrize('methods', [['omp'], None])
def test_gui_worker_prepares_optional_inputs_without_running_estimator(saved_data, tmp_path, methods):
    import time
    from cgap_sbl_jadce.gui.jobs import Jobs
    from cgap_sbl_jadce.recovery import load_recoveries
    config = tmp_path / 'source.yaml'; save_yaml(config, saved_data)
    add_condition(saved_data, 'omp', PAIR)
    request = {'kind': 'task', 'task': 'communication', 'action': 'run', 'config': str(config),
               'conditions': [list(PAIR)]}
    if methods is not None:
        request['methods'] = methods
    jobs = Jobs(tmp_path)
    record = jobs.start(request)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = jobs.records()[0]
            if state['status'] not in {'queued', 'running'}:
                break
            time.sleep(.05)
        assert state['status'] == 'completed', jobs.log(record['id'])[0]
        assert not input_reasons(saved_data, 'omp', PAIR)
        validate_communication_dataset(saved_data, 'tuning', [PAIR])
        assert not run_path(saved_data, 'omp').exists()
        assert not load_recoveries(config, 'communication')
        assert not list(task_dir(saved_data['run']['output_dir'], 'communication').rglob('.checkpoints/*/*.npz'))
    finally:
        if jobs.busy():
            jobs.control(record['id'], 'stop')
        jobs.children[record['id']].wait(timeout=10)


def test_running_optional_work_does_not_change_overall_completion(saved_data, tmp_path):
    from types import SimpleNamespace
    from cgap_sbl_jadce.gui.workflow import card_state, combined
    from cgap_sbl_jadce.gui.jobs import config_digest
    add_condition(saved_data, 'omp', PAIR)
    estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[(7, 8)])
    config = tmp_path / 'source.yaml'; save_yaml(config, saved_data)
    request = {'kind': 'task', 'task': 'omp', 'action': 'run', 'config': str(config), 'conditions': [list(PAIR)]}
    record = {'id': 'optional', 'status': 'running', 'request': request, 'config_scope': 2,
              'config_digest': config_digest(config, request)}
    jobs = SimpleNamespace(task_job=lambda *a: record,
                           log=lambda *a: ('', {'done': 1, 'total': 1, 'label': 'Optional realization'}))
    state = card_state(saved_data, config, 'omp', jobs)
    assert state.status == 'running' and state.fraction == pytest.approx(1 / 3)
    from cgap_sbl_jadce.gui.workflow import CardState
    parent = combined([CardState('completed', fraction=1.), CardState('running', fraction=1 / 3)])
    assert parent.fraction == pytest.approx(2 / 3)
