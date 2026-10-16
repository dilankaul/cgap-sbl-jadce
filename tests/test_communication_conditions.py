import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from cgap_sbl_jadce.additional_conditions import add_condition
from cgap_sbl_jadce.communication_dataset import validate_communication_dataset
from cgap_sbl_jadce.experiments import estimators, pipeline
from cgap_sbl_jadce.gui.communication_conditions import condition_rows
from cgap_sbl_jadce.io import save_json, save_npz, save_yaml
from cgap_sbl_jadce.paths import task_dir
from cgap_sbl_jadce.settings import grouped_config

PAIR = (9, 10.)


def versions(folder):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in folder.rglob('*') if path.is_file()}


def test_communication_prepares_shared_pair_in_both_splits_preserving_completed_work(saved_data, monkeypatch):
    estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[(7, 8)])
    out = saved_data['run']['output_dir']
    add_condition(saved_data, 'cosamp', PAIR)
    pipeline.communication_stage(saved_data, False, conditions=[PAIR], methods=['omp'])
    estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[PAIR])
    algorithm = versions(task_dir(out, 'omp'))
    communication = task_dir(out, 'communication')
    base = {path: value for path, value in versions(communication).items() if path.suffix == '.npz'}
    rows, issues = condition_rows(saved_data)
    assert not issues
    optional = next(row for row in rows if row['Type'] == 'Optional')
    assert optional['Tuning inputs'] == 'Missing' and optional['Evaluation inputs'] == 'Ready'
    pipeline.communication_stage(saved_data, False, conditions=[PAIR])
    for split in ('tuning', 'evaluation'):
        validate_communication_dataset(saved_data, split, [PAIR])
    rows, issues = condition_rows(saved_data)
    assert not issues
    optional = next(row for row in rows if row['Type'] == 'Optional')
    assert optional['Tuning inputs'] == optional['Evaluation inputs'] == 'Ready'
    assert versions(task_dir(out, 'omp')) == algorithm
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in base.items())
    assert estimators.completed_methods(saved_data, [(7, 8)]) == ['omp']
    assert estimators.completed_methods(saved_data, [PAIR]) == ['omp']
    from cgap_sbl_jadce.models import realizations
    monkeypatch.setattr(realizations, 'communication_realization', lambda *a: pytest.fail('Repeated completed input generation'))
    saved = versions(communication)
    pipeline.communication_stage(saved_data, False, conditions=[PAIR])
    assert versions(communication) == saved


def test_selected_preparation_fills_missing_setup_tuning_without_replacing_base_archives(saved_data):
    import numpy as np
    root = task_dir(saved_data['run']['output_dir'], 'communication')
    for q in range(2):
        path = root / 'tuning' / f'{q:04d}' / 'observations.npz'
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files if key.startswith('L5_snr12_')}
        save_npz(path, **arrays)
    path = root / 'summary.json'
    summary = json.loads(path.read_text())
    summary['tuning']['definition']['conditions'] = [[5, 12.]]
    save_json(path, summary)
    base = {path: value for path, value in versions(root).items() if path.suffix == '.npz'}
    pipeline.communication_stage(saved_data, False, conditions=[(7, 8)])
    validate_communication_dataset(saved_data, 'tuning', [(7, 8)])
    assert (root / 'tuning/additional/L0007_snr8dB/0000/inputs.npz').exists()
    assert not (root / 'evaluation/additional').exists()
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in base.items())


def test_both_split_preparation_resumes_without_repeating_completed_split(saved_data, monkeypatch):
    from cgap_sbl_jadce.models import realizations
    add_condition(saved_data, 'omp', PAIR)
    generate = realizations.communication_realization
    calls = []
    def interrupted(activity, seed, q, *args):
        calls.append((seed, q))
        if len(calls) == 4:
            raise RuntimeError('interrupted during tuning')
        return generate(activity, seed, q, *args)
    monkeypatch.setattr(realizations, 'communication_realization', interrupted)
    with pytest.raises(RuntimeError, match='during tuning'):
        pipeline.communication_stage(saved_data, False, conditions=[PAIR])
    validate_communication_dataset(saved_data, 'evaluation', [PAIR])
    rows, _ = condition_rows(saved_data)
    optional = next(row for row in rows if row['Type'] == 'Optional')
    assert optional['Evaluation inputs'] == 'Ready' and optional['Tuning inputs'] == 'Missing'
    calls.clear()
    monkeypatch.setattr(realizations, 'communication_realization',
        lambda activity, seed, q, *args: (calls.append((seed, q)), generate(activity, seed, q, *args))[1])
    pipeline.communication_stage(saved_data, False, conditions=[PAIR], resume=True)
    assert calls == [(saved_data['seeds']['communication_tuning'], 1)]
    validate_communication_dataset(saved_data, 'tuning', [PAIR])


def test_algorithm_added_condition_appears_in_communication_and_launches_input_only_job(saved_data, tmp_path, monkeypatch):
    from cgap_sbl_jadce.gui.jobs import Jobs
    config = tmp_path / 'source.yaml'
    save_yaml(config, grouped_config(saved_data))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    requests = []
    monkeypatch.setattr(Jobs, 'start', lambda self, request: requests.append(request) or {'id': 'test'})
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    app.button(key='open_algorithm_omp').click().run()
    next(w for w in app.number_input if w.label == 'Additional pilot length L').set_value(9)
    next(w for w in app.number_input if w.label == 'Additional SNR (dB)').set_value(10.)
    next(b for b in app.button if b.label == 'Add').click().run()
    app.button(key='go_back').click().run()
    app.button(key='open_communication_run').click().run()
    assert not app.exception
    table = next(w.value for w in app.dataframe if 'Tuning inputs' in w.value.columns)
    row = table[table.Type.eq('Optional')].iloc[0]
    assert row['Pilot length L'] == 9 and row['SNR (dB)'] == 10.
    assert row['Tuning inputs'] == row['Evaluation inputs'] == 'Missing'
    selector = next(w for w in app.multiselect if w.label == 'Communication conditions')
    assert PAIR in selector.options or PAIR in selector.value
    next(b for b in app.button if b.label == 'Complete selected inputs').click().run()
    assert not app.exception
    assert len(requests) == 1
    assert requests[0]['task'] == 'communication' and requests[0]['conditions'] == [[9, 10.]]
    assert 'methods' not in requests[0]
