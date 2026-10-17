import os
from pathlib import Path
import time
from types import SimpleNamespace

from cgap_sbl_jadce.gui.jobs import Jobs, process_token
from cgap_sbl_jadce.gui.workflow import refresh_results
from cgap_sbl_jadce.io import save_json, save_yaml
from cgap_sbl_jadce.paths import run_path, task_dir


def test_saved_condition_changes_refresh_while_worker_status_stays_running(tmp_path):
    cfg = {'run': {'output_dir': str(tmp_path)}}
    updates = []
    st = SimpleNamespace(session_state={}, rerun=lambda: updates.append(True))
    records = [{'id': 'all_conditions', 'status': 'running'}]
    refresh_results(st, cfg, records)
    refresh_results(st, cfg, records)
    assert updates == []
    for pair in [(25, 12.), (30, 12.)]:
        path = run_path(cfg, 'omp', 'summary.json', pair)
        save_json(path, {'status': 'running'})
        refresh_results(st, cfg, records)
        count = len(updates)
        save_json(path, {'status': 'complete'})
        refresh_results(st, cfg, records)
        assert len(updates) == count + 1
        refresh_results(st, cfg, records)
        assert len(updates) == count + 1
    assert records[0]['status'] == 'running'


def test_full_page_render_updates_revision_without_interrupting_navigation(tmp_path):
    cfg = {'run': {'output_dir': str(tmp_path)}}
    updates = []
    st = SimpleNamespace(session_state={}, rerun=lambda: updates.append(True))
    refresh_results(st, cfg, [])
    save_json(run_path(cfg, 'omp', 'summary.json', (25, 12.)), {'status': 'complete'})
    refresh_results(st, cfg, [], rerun=False)
    refresh_results(st, cfg, [])
    assert updates == []


def test_poll_checks_result_metadata_without_reading_arrays_or_unit_checkpoints(tmp_path, monkeypatch):
    cfg = {'run': {'output_dir': str(tmp_path)}}
    updates = []
    st = SimpleNamespace(session_state={}, rerun=lambda: updates.append(True))
    refresh_results(st, cfg, [])
    checkpoint = run_path(cfg, 'omp', condition=(25, 12.)) / '.checkpoints/key/0000.npz'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint')
    communication = task_dir(tmp_path, 'communication') / 'evaluation/0000/channels.npz'
    communication.parent.mkdir(parents=True)
    communication.write_bytes(b'channels')
    monkeypatch.setattr(Path, 'read_bytes', lambda path: (_ for _ in ()).throw(AssertionError('Read array data')))
    refresh_results(st, cfg, [])
    assert updates == []
    selection = task_dir(tmp_path, 'glc') / 'tuning/L0025_snr12dB/selected.json'
    save_json(selection, {'status': 'complete'})
    refresh_results(st, cfg, [])
    assert len(updates) == 1
    selection.unlink()
    refresh_results(st, cfg, [])
    assert len(updates) == 2
    summary = selection.with_name('summary.json')
    save_json(summary, {'status': 'started'})
    refresh_results(st, cfg, [])
    assert len(updates) == 3
    save_json(summary, {'status': 'complete'})
    refresh_results(st, cfg, [])
    assert len(updates) == 4


def test_run_all_page_shows_each_completed_condition_before_worker_finishes(saved_data, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from cgap_sbl_jadce.experiments.estimators import run_estimator_stage
    cfg = saved_data
    config = tmp_path / 'experiment.yaml'
    save_yaml(config, cfg)
    jobs = Jobs(tmp_path)
    folder = jobs.folder / 'run_all'
    folder.mkdir()
    save_json(folder / 'job.json', {'id': 'run_all', 'status': 'running', 'created': time.time(),
        'pid': os.getpid(), 'process_token': process_token(os.getpid()),
        'request': {'kind': 'task', 'config': str(config), 'task': 'omp', 'action': 'run'}})
    save_json(folder / 'control.json', {'command': 'run'})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    app.button(key='open_algorithm_omp').click().run()
    assert not app.exception
    for count, pair in enumerate([(7, 8.), (7, 12.), (5, 12.)], 1):
        previous = app.session_state['_workflow_revision']
        run_estimator_stage(cfg, 'omp', False, conditions=[pair])
        app.run()
        assert not app.exception
        assert app.session_state['_workflow_revision'] != previous
        assert jobs.records()[0]['status'] == 'running'
        assert any(element.proto.text == f'{count}/3 conditions completed' for element in app.get('progress'))
        assert sum(item.value == 'Completed' for item in app.markdown) == count
