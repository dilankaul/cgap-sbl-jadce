from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.settings import grouped_config, tuning_pairs, tuning_settings
from cgap_sbl_jadce.gui.setups import update_setup, create_setup, change_impact
from cgap_sbl_jadce.gui.workflow import numerical_state
from cgap_sbl_jadce.gui.jobs import config_digest
from cgap_sbl_jadce.gui.inputs import input_signature
from cgap_sbl_jadce.experiments import estimators, pipeline
from cgap_sbl_jadce.experiments.readiness import blocking_reasons
from cgap_sbl_jadce.communication_dataset import validate_communication_dataset
from cgap_sbl_jadce.paths import run_path, result_path
from cgap_sbl_jadce.recovery import begin_recovery, validate_recovery
from cgap_sbl_jadce.io import save_yaml, save_json


def test_grouping_preserves_values_and_independent_parameter_ownership(saved_data):
    grouped = grouped_config(saved_data)
    assert 'tuning' not in grouped and 'alpha_values' not in grouped['algorithms']['sbl']['tuning']
    assert estimators.evaluation_conditions(grouped) == estimators.evaluation_conditions(saved_data)
    for method in ('omp', 'cosamp', 'sbl', 'anc'):
        if method in {'sbl', 'anc'}:
            assert estimators.tuning_definition(grouped, method) == estimators.tuning_definition(saved_data, method)
        assert estimators._anc_grid(grouped) == estimators._anc_grid(saved_data)
    grouped['algorithms']['future'] = {'solver': {'K': 17}, 'tuning': {'alpha_values': [0, 3]}}
    assert grouped_config(grouped)['algorithms']['future'] == grouped['algorithms']['future']


def test_anc_edit_invalidates_only_anc_and_reverting_restores_results(saved_data):
    cfg = grouped_config(saved_data)
    for method in ('sbl', 'anc', 'gl'):
        estimators.tune_estimator_stage(cfg, method, False)
    for method in estimators.ESTIMATORS:
        estimators.run_estimator_stage(cfg, method, False, conditions=[[7, 8]])
    before = run_path(cfg, 'anc', 'summary.json', (7, 8)).read_bytes()
    cfg['algorithms']['anc']['tuning']['alpha_high'] = 4.
    assert set(estimators.completed_methods(cfg, [[7, 8]])) == {'omp', 'cosamp', 'sbl', 'gl'}
    assert numerical_state(cfg, 'anc', 'tune').status == 'incompatible'
    assert numerical_state(cfg, 'anc').status == 'incompatible'
    assert blocking_reasons(cfg, 'anc', conditions=[[7, 8]])
    cfg['algorithms']['anc']['tuning']['alpha_high'] = 2.
    assert set(estimators.completed_methods(cfg, [[7, 8]])) == set(estimators.ESTIMATORS)
    assert not blocking_reasons(cfg, 'anc', conditions=[[7, 8]])
    assert run_path(cfg, 'anc', 'summary.json', (7, 8)).read_bytes() == before


def test_added_conditions_keep_finished_pairs_without_repeated_solves(saved_data, monkeypatch):
    cfg = grouped_config(saved_data)
    estimators.run_estimator_stage(cfg, 'omp', False, conditions=[[7, 8]])
    output = run_path(cfg, 'omp', 'per_sample.csv', (7, 8)).read_bytes()
    cfg['evaluation']['pilot_length_stop'] = 9
    assert estimators.completed_methods(cfg, [[7, 8]]) == ['omp']
    assert not blocking_reasons(cfg, 'omp', conditions=[[7, 8]])
    assert blocking_reasons(cfg, 'omp', conditions=[[9, 12]])
    pipeline.communication_stage(cfg, False)
    assert estimators.completed_methods(cfg, [[7, 8]]) == ['omp']
    spec = estimators.ESTIMATORS['omp']
    monkeypatch.setitem(estimators.ESTIMATORS, 'omp', replace(spec, solve=lambda *a, **k: pytest.fail('Repeated finished condition')))
    estimators.run_estimator_stage(cfg, 'omp', False, conditions=[[7, 8]], resume=True)
    assert run_path(cfg, 'omp', 'per_sample.csv', (7, 8)).read_bytes() == output


def test_sweep_subset_preserves_completed_measurements(saved_data):
    cfg = grouped_config(saved_data)
    estimators.run_estimator_stage(cfg, 'omp', False, conditions=[[7, 12]])
    cfg['evaluation'].update(pilot_length_start=7, pilot_length_stop=7, snr_start=12., snr_stop=12.)
    assert {value[:2] for value in estimators.evaluation_conditions(cfg)} == {(7, 12.)}
    assert estimators.completed_methods(cfg) == ['omp']
    assert numerical_state(cfg, 'omp').status == 'completed'
    comparison = estimators.comparison_stage(cfg, False, methods=['omp'])
    assert comparison['metric_rows'] == 4


def test_separate_algorithm_tuning_policies_and_reference_conditions(saved_data):
    cfg = grouped_config(saved_data)
    cfg['algorithms']['sbl']['tuning']['selection_mode'] = 'per_condition'
    cfg['algorithms']['anc']['tuning'].update(reference_pilot_length=7, reference_snr_db=8.)
    pipeline.communication_stage(cfg, False)
    assert tuning_pairs(cfg, 'sbl') == [(7, 8.), (7, 12.), (5, 12.)]
    assert tuning_pairs(cfg, 'anc') == [(7, 8.)]
    estimators.tune_estimator_stage(cfg, 'anc', False)
    assert estimators.load_selection(cfg, 'anc')['definition']['condition'] == [7, 8.]
    estimators.tune_estimator_stage(cfg, 'sbl', False, conditions=[[7, 12]])
    assert not blocking_reasons(cfg, 'sbl', conditions=[[7, 12]])
    assert blocking_reasons(cfg, 'sbl', conditions=[[7, 8]])


def test_figure_and_unrelated_solver_edits_leave_omp_and_recovery_compatible(saved_data, tmp_path):
    cfg = grouped_config(saved_data)
    config = tmp_path / 'settings.yaml'
    save_yaml(config, cfg)
    request = {'kind': 'task', 'task': 'omp', 'action': 'run', 'conditions': [[7, 8]], 'config': str(config)}
    begin_recovery(request)
    digest, signature = config_digest(config, request), input_signature(request, portable=True)
    cfg['figures']['evaluation_indices'] = [1]
    cfg['algorithms']['anc']['solver']['K'] += 1
    cfg['evaluation']['pilot_length_stop'] = 9
    update_setup(config, cfg)
    config.write_text(config.read_text() + '\n# Formatting changes are harmless.\n')
    assert config_digest(config, request) == digest
    assert input_signature(request, portable=True) == signature
    validate_recovery(request)
    cfg['system']['M'] += 1
    update_setup(config, cfg)
    with pytest.raises(ValueError, match='Configuration changed'):
        validate_recovery(request)
    cfg['system']['M'] -= 1
    update_setup(config, cfg)
    validate_recovery(request)


def test_existing_setup_edit_preserves_identity_results_and_recovery_files(saved_data, tmp_path):
    cfg = grouped_config(saved_data)
    path = tmp_path / 'source.yaml'
    save_yaml(path, cfg)
    result = run_path(cfg, 'omp', 'summary.json', (7, 8))
    save_json(result, {'status': 'example'})
    cfg['algorithms']['anc']['tuning']['beta_high'] = .2
    assert update_setup(path, cfg) == ['algorithms.anc.tuning.beta_high']
    assert json.loads(result.read_text()) == {'status': 'example'}
    bad = deepcopy(cfg); bad['run']['output_dir'] = str(tmp_path / 'other')
    with pytest.raises(ValueError, match='identity'):
        update_setup(path, bad)


def test_active_settings_guard_allows_unrelated_changes_and_requires_pause(saved_data, tmp_path):
    config = tmp_path / 'source.yaml'
    cfg = grouped_config(saved_data); save_yaml(config, cfg)
    request = {'kind': 'task', 'task': 'omp', 'action': 'run', 'config': str(config), 'conditions': [[7, 8]]}
    job = {'status': 'running', 'request': request}
    jobs = SimpleNamespace(records=lambda: [job])
    cfg['algorithms']['anc']['tuning']['beta_high'] = .2
    update_setup(config, cfg, jobs)
    cfg['system']['M'] += 1
    with pytest.raises(ValueError, match='Pause'):
        update_setup(config, cfg, jobs)
    job['status'] = 'paused'
    update_setup(config, cfg, jobs)


def test_preparation_seed_edits_block_successors_and_restore(saved_data):
    cfg = deepcopy(saved_data)
    cfg['seeds']['activity_evaluation'] += 1
    assert numerical_state(cfg, 'activity').status == 'incompatible'
    assert blocking_reasons(cfg, 'communication')
    assert blocking_reasons(cfg, 'omp', conditions=[[7, 8]])
    cfg['seeds']['activity_evaluation'] -= 1
    assert not blocking_reasons(cfg, 'communication')
    assert not blocking_reasons(cfg, 'omp', conditions=[[7, 8]])


def test_setup_and_task_editors_share_settings_without_running_tasks(saved_data, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    cfg = grouped_config(saved_data)
    config = tmp_path / 'source.yaml'; save_yaml(config, cfg)
    monkeypatch.chdir(tmp_path); monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    next(b for b in app.sidebar.button if b.label == 'Experiment setups').click().run()
    assert not any('All pilot lengths' in str(widget.options) for widget in app.radio)
    assert len([w for w in app.number_input if w.label == 'Alpha start']) == 1
    next(w for w in app.number_input if w.key.endswith('_anc_solver_K')).set_value(4)
    assert not app.error, [error.value for error in app.error]
    assert load_config(config)['algorithms']['anc']['solver']['K'] == 3
    next(b for b in app.button if b.label == 'Save changes').click().run()
    assert load_config(config)['algorithms']['anc']['solver']['K'] == 4
    app.button(key='go_back').click().run()
    app.button(key='open_algorithm_anc').click().run()
    app.button(key='open_anc_tune').click().run()
    assert not app.exception
    assert not [w for w in app.number_input if w.label == 'Alpha start']
    assert len([w for w in app.number_input if w.label == 'Alpha low']) == 1
    next(w for w in app.number_input if w.key.endswith('_anc_solver_K')).set_value(3)
    next(b for b in app.button if b.label == 'Save changes').click().run()
    assert not app.exception
    assert load_config(config)['algorithms']['anc']['solver']['K'] == 3
    assert not list((tmp_path / '.gui').glob('*/job.json'))


def test_calibration_and_activity_compatibility_recover_after_editing_back(tmp_path):
    from cgap_sbl_jadce.compatibility import preparation_settings
    from cgap_sbl_jadce.config import resolve_activity_kappa
    cfg = grouped_config(load_config('tests/fixtures/small_experiment.yaml'))
    cfg['run'].update(output_dir=str(tmp_path), num_tuning_samples=2, num_evaluation_samples=2)
    cfg['figures'] = {'tuning_indices': [], 'evaluation_indices': []}
    summary = {**{key: cfg['system'][key] for key in ('N', 'R')}, **{key: cfg['activity'][key] for key in ('V', 'D')},
               'target_mean_active': cfg['system']['S'], 'seed': cfg['seeds']['kappa_calibration'],
               'closest_grid_kappa': 3.78, 'status': 'success', 'compatibility': preparation_settings(cfg, 'kappa_calibration')}
    save_json(result_path(tmp_path, 'kappa_calibration', 'summary.json'), summary)
    pipeline.generate_activity_stage(cfg, False)
    before = result_path(tmp_path, 'activity', 'evaluation_activity.npz').read_bytes()
    cfg['kappa_calibration']['kappa_refinement_steps'][-1] = .001
    with pytest.raises(ValueError, match='kappa_refinement_steps'):
        resolve_activity_kappa(cfg)
    assert numerical_state(cfg, 'kappa_calibration').status == 'incompatible'
    assert numerical_state(cfg, 'activity').status == 'incompatible'
    assert blocking_reasons(cfg, 'communication')
    cfg['kappa_calibration']['kappa_refinement_steps'][-1] = .005
    assert numerical_state(cfg, 'activity').status == 'completed'
    cfg['system']['M'] += 1
    assert numerical_state(cfg, 'kappa_calibration').status == 'completed'
    assert numerical_state(cfg, 'activity').status == 'completed'
    assert result_path(tmp_path, 'activity', 'evaluation_activity.npz').read_bytes() == before


def test_incompatible_preparation_cannot_overwrite_existing_communication_summary(saved_data):
    cfg = grouped_config(saved_data)
    path = result_path(cfg['run']['output_dir'], 'communication', 'summary.json')
    before = path.read_bytes()
    cfg['system']['N'] += 1
    with pytest.raises(ValueError, match='incompatible'):
        pipeline.communication_stage(cfg, False)
    assert path.read_bytes() == before


def test_saved_tuning_graphs_remain_available_after_policy_changes(saved_data, tmp_path, monkeypatch):
    from cgap_sbl_jadce.gui.previews import stored_tuning_options, render_preview
    from cgap_sbl_jadce.gui.charts import interactive_figures, source_version
    cfg = grouped_config(saved_data)
    estimators.tune_estimator_stage(cfg, 'sbl', False)
    cfg['algorithms']['sbl']['tuning'].update(selection_mode='per_condition', reference_pilot_length=9)
    config = tmp_path / 'settings.yaml'; save_yaml(config, cfg)
    assert stored_tuning_options(cfg, 'sbl') == [('reference', 5, 12.)]
    request = {'config': str(config), 'category': 'sbl_tune', 'tuning_layout': 'reference', 'condition': [5, 12]}
    monkeypatch.setattr(estimators, 'tune_estimator_stage', lambda *a, **k: pytest.fail('Graph ran tuner'))
    assert interactive_figures(request)
    assert source_version(request)
    exports = render_preview(request, tmp_path / 'export')
    assert {path.suffix for path in exports} == {'.png', '.pdf'}
    assert all(path.exists() for path in exports)


def test_preparation_without_compatibility_metadata_is_rejected(saved_data):
    path = result_path(saved_data['run']['output_dir'], 'activity', 'summary.json')
    summary = json.loads(path.read_text())
    summary.pop('compatibility')
    save_json(path, summary)
    before = result_path(saved_data['run']['output_dir'], 'communication', 'summary.json').read_bytes()
    with pytest.raises(ValueError, match='metadata missing'):
        pipeline.communication_stage(saved_data, False)
    assert result_path(saved_data['run']['output_dir'], 'communication', 'summary.json').read_bytes() == before
