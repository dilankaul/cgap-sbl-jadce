from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.gui.setups import create_setup, setup_files, validate_setup
from cgap_sbl_jadce.experiments import pipeline, estimators
from cgap_sbl_jadce.experiments.readiness import blocking_reasons
from cgap_sbl_jadce.paths import result_path, tuning_path
from cgap_sbl_jadce.provenance import save_task_snapshot


def test_new_setups_copy_defaults_and_keep_one_stable_config(tmp_path):
    base = load_config('tests/fixtures/small_experiment.yaml')
    original = deepcopy(base)
    path = create_setup(tmp_path, 'Environment A', base)
    other = create_setup(tmp_path, 'Environment B', base)
    assert base == original
    assert load_config(path)['run']['output_dir'] != load_config(other)['run']['output_dir']
    before = path.read_bytes()
    import yaml
    assert yaml.safe_load(before)['run']['output_dir'] == 'results/environment-a'
    cfg = load_config(path)
    cfg['task_runtime'] = {'input_config': str(path)}
    cfg['activity']['kappa'] = 7
    save_task_snapshot(cfg, 'activity', status='complete')
    assert path.read_bytes() == before
    assert list(path.parent.glob('*.yaml')) == [path]
    assert path in setup_files(tmp_path, path)
    with pytest.raises(FileExistsError):
        create_setup(tmp_path, 'Environment A', base)
    with pytest.raises(ValueError):
        create_setup(tmp_path, '../../outside', base)


def test_condition_policy_generates_matching_tuning_conditions(tmp_path):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['tuning']['selection_mode'] = 'per_condition'
    path = create_setup(tmp_path, 'Condition tuning', cfg)
    saved = load_config(path)
    expected = {tuple(pair[:2]) for pair in estimators.evaluation_conditions(saved)}
    from cgap_sbl_jadce.settings import tuning_pairs
    actual = set(tuning_pairs(saved))
    assert expected <= actual
    for mutation in [lambda c: c['system'].update(S=c['system']['N'] + 1),
                     lambda c: c['evaluation'].update(pilot_length_stop=-5),
                     lambda c: c['seeds'].update(activity_tuning=c['seeds']['activity_evaluation']),
                     lambda c: c['figures'].update(evaluation_indices=[999])]:
        bad = deepcopy(saved)
        mutation(bad)
        with pytest.raises(ValueError):
            validate_setup(bad)


def test_incremental_conditions_preserve_scores_and_explicit_reruns_recalculate(saved_data, monkeypatch):
    spec = estimators.ESTIMATORS['omp']
    calls = []
    def solve(r, *args, **kwargs):
        calls.append(r.Theta.shape[0])
        return spec.solve(r, *args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=solve))
    first = estimators.run_estimator_stage(saved_data, 'omp', show_progress=False, conditions=[[7, 8]])
    assert first['status'] == 'partial' and len(calls) == 2
    path = result_path(saved_data['run']['output_dir'], 'omp', 'per_sample.csv')
    before = pd.read_csv(path)
    estimators.run_estimator_stage(saved_data, 'omp', show_progress=False, conditions=[[5, 12]])
    assert len(calls) == 4
    after = pd.read_csv(path)
    pd.testing.assert_frame_equal(before, after[after.snr_db == 8].reset_index(drop=True))
    estimators.run_estimator_stage(saved_data, 'omp', show_progress=False, conditions=[[7, 8]])
    assert len(calls) == 6  # A fresh rerun recalculates the selected condition.
    final = estimators.run_estimator_stage(saved_data, 'omp', show_progress=False)
    assert final['status'] == 'complete' and len(calls) == 12
    assert not pd.read_csv(path).duplicated(['q', 'L', 'snr_db', 'sweep']).any()


def per_condition_data(cfg):
    cfg['tuning']['selection_mode'] = 'per_condition'
    cfg['communication'] = {'tuning_conditions': [list(pair) for pair in dict.fromkeys(
        (L, snr) for L, snr, _ in estimators.evaluation_conditions(cfg))]}
    pipeline.communication_stage(cfg, show_progress=False)
    return cfg


@pytest.mark.parametrize('method', ['sbl', 'anc'])
def test_per_condition_tuning_and_runs_need_only_requested_selections(saved_data, method, monkeypatch):
    cfg = per_condition_data(saved_data)
    spec = estimators.ESTIMATORS[method]
    selected = estimators.tune_estimator_stage(cfg, method, False, conditions=[[7, 8]])
    assert selected['definition']['condition'] == [7, 8]
    assert tuning_path(cfg, method, 'selected.json', (7, 8)).exists()
    assert not tuning_path(cfg, method, 'selected.json', (5, 12)).exists()
    assert not blocking_reasons(cfg, spec.task, 'run', [[7, 8]])
    assert blocking_reasons(cfg, spec.task, 'run', [[5, 12]])
    estimators.run_estimator_stage(cfg, method, False, conditions=[[7, 8]])
    before = pd.read_csv(result_path(cfg['run']['output_dir'], method, 'per_sample.csv'))
    estimators.tune_estimator_stage(cfg, method, False, conditions=[[5, 12]])
    estimators.run_estimator_stage(cfg, method, False, conditions=[[5, 12]])
    after = pd.read_csv(result_path(cfg['run']['output_dir'], method, 'per_sample.csv'))
    pd.testing.assert_frame_equal(before, after[after.snr_db == 8].reset_index(drop=True))
    monkeypatch.setitem(estimators.ESTIMATORS, method, replace(spec, solve=lambda *a, **k: pytest.fail('Repeated finished solve')))
    estimators.run_estimator_stage(cfg, method, False, conditions=[[7, 8]], resume=True)


def test_algorithms_and_selected_condition_comparison_have_no_order_dependency(saved_data):
    assert not blocking_reasons(saved_data, 'omp', conditions=[[7, 8]])
    assert not blocking_reasons(saved_data, 'cosamp', conditions=[[7, 8]])
    assert not blocking_reasons(saved_data, 'sbl', action='tune')
    estimators.tune_estimator_stage(saved_data, 'sbl', False)
    assert not blocking_reasons(saved_data, 'sbl', conditions=[[7, 8]])
    # SBL finishes before either greedy algorithm; ANC is independent as well.
    estimators.run_estimator_stage(saved_data, 'sbl', False, conditions=[[7, 8]])
    estimators.tune_estimator_stage(saved_data, 'anc', False)
    for method in ('anc', 'cosamp', 'omp'):
        estimators.run_estimator_stage(saved_data, method, False, conditions=[[7, 8]])
    assert not blocking_reasons(saved_data, 'comparison', conditions=[[7, 8]])
    assert blocking_reasons(saved_data, 'comparison', conditions=[[5, 12]])
    summary = estimators.comparison_stage(saved_data, False, conditions=[[7, 8]])
    assert summary['metric_rows'] == 8


def test_changed_condition_selection_cannot_reuse_its_previous_scores(saved_data, monkeypatch):
    cfg = per_condition_data(saved_data)
    estimators.tune_estimator_stage(cfg, 'sbl', False, conditions=[[7, 8], [5, 12]])
    estimators.run_estimator_stage(cfg, 'sbl', False, conditions=[[7, 8], [5, 12]])
    path = tuning_path(cfg, 'sbl', 'selected.json', (7, 8))
    selection = json.loads(path.read_text())
    selection['parameters']['tau'] = .2 if selection['parameters']['tau'] != .2 else .1
    path.write_text(json.dumps(selection))
    spec = estimators.ESTIMATORS['sbl']
    calls = []
    def solve(r, *args, **kwargs):
        calls.append(r.Theta.shape[0])
        return spec.solve(r, *args, **kwargs)
    monkeypatch.setitem(estimators.ESTIMATORS, 'sbl', replace(spec, solve=solve))
    estimators.run_estimator_stage(cfg, 'sbl', False, conditions=[[7, 8]])
    assert calls == [7, 7]
    df = pd.read_csv(result_path(cfg['run']['output_dir'], 'sbl', 'per_sample.csv'))
    assert len(df) == 4


def test_condition_tune_dispatch_and_pdf_export_use_selected_scope(saved_data, tmp_path):
    from cgap_sbl_jadce.tasks import run_task
    from cgap_sbl_jadce.gui.previews import render_preview
    from cgap_sbl_jadce.io import save_yaml
    cfg = per_condition_data(saved_data)
    paths = run_task('sbl', cfg, action='tune', show_progress=False, conditions=[[7, 8]])
    assert len(paths) == 1 and 'L0007_snr8dB' in str(paths[0])
    config = tmp_path / 'input.yaml'
    save_yaml(config, cfg)
    exported = render_preview({'config': str(config), 'category': 'sbl_tune', 'condition': [7, 8]}, tmp_path / 'export')
    assert {path.suffix for path in exported} == {'.png', '.pdf'}
    assert all('L0007_snr8dB' in str(path) for path in exported)
