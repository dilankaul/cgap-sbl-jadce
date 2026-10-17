import json
import os
from pathlib import Path
import signal
import time

import pytest

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.gui.jobs import Jobs, process_token
from cgap_sbl_jadce.gui.previews import render_preview, sample_options, missing_preview_inputs
from cgap_sbl_jadce.io import save_json, save_yaml
from cgap_sbl_jadce.paths import result_path, task_dir


def wait_status(jobs, expected, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = jobs.records()[0]
        if record['status'] in expected:
            return record
        time.sleep(.05)
    logs, progress = jobs.log(record['id'])
    pytest.fail(f"Worker did not reach {expected}: {record}\n{logs}")


def test_worker_pause_survives_controller_restart_and_rejects_second_job(saved_data, tmp_path):
    config = tmp_path / 'experiment.yaml'
    save_yaml(config, saved_data)
    jobs = Jobs(tmp_path)
    request = {'kind': 'task', 'task': 'omp', 'action': 'run', 'config': str(config)}
    record = jobs.start(request)
    try:
        jobs.control(record['id'], 'pause')
        wait_status(jobs, {'paused'})
        reopened = Jobs(tmp_path)
        assert reopened.active()['id'] == record['id']
        with pytest.raises(ValueError, match='already has paused work'):
            reopened.start(request)

        original = config.read_text()
        changed = load_config(config)
        changed['system']['M'] += 1
        save_yaml(config, changed)
        with pytest.raises(ValueError, match='Configuration changed'):
            reopened.resume(record['id'])
        config.write_text(original)
        resumed = reopened.resume(record['id'])
        assert (resumed['id'], resumed['pid']) == (record['id'], record['pid'])
        completed = wait_status(reopened, {'completed', 'failed'})
        assert completed['status'] == 'completed', reopened.log(record['id'])[0]
        assert Path(completed['paths'][0]).exists()
        with pytest.raises(ValueError, match='can resume'):
            reopened.resume(record['id'])
        assert result_path(saved_data['run']['output_dir'], 'omp', 'runtime_summary.csv').exists()
    finally:
        if jobs.active():
            jobs.control(record['id'], 'stop')
        jobs.children[record['id']].wait(timeout=10)


def test_killed_worker_detected_and_resume_uses_committed_checkpoints(saved_data, tmp_path):
    config = tmp_path / 'experiment.yaml'
    save_yaml(config, saved_data)
    jobs = Jobs(tmp_path)
    record = jobs.start({'kind': 'task', 'task': 'omp', 'action': 'run', 'config': str(config)})
    try:
        checkpoint_folder = task_dir(saved_data['run']['output_dir'], 'omp') / 'run'
        deadline = time.monotonic() + 20
        while not list(checkpoint_folder.glob('*/.checkpoints/*/0000.npz')):
            if time.monotonic() > deadline:
                pytest.fail('No evaluation checkpoint written')
            time.sleep(.01)
        jobs.control(record['id'], 'pause')
        paused = wait_status(jobs, {'paused', 'completed'})
        assert paused['status'] == 'paused'
        os.kill(record['pid'], signal.SIGKILL)
        jobs.children[record['id']].wait(timeout=10)
        reopened = Jobs(tmp_path)
        assert reopened.records()[0]['status'] == 'interrupted'
        original = config.read_text()
        changed = load_config(config)
        changed['system']['M'] += 1
        save_yaml(config, changed)
        with pytest.raises(ValueError, match='Configuration changed'):
            reopened.resume(record['id'])
        config.write_text(original)
        resumed = reopened.resume(record['id'])
        assert resumed['id'] != record['id']
        assert resumed['request'] == record['request']
        finished = wait_status(reopened, {'completed', 'failed'})
        assert finished['status'] == 'completed', reopened.log(resumed['id'])[0]
        reopened.children[resumed['id']].wait(timeout=10)
    finally:
        if jobs.active():
            jobs.control(jobs.active()['id'], 'stop')


def test_stop_does_not_delete_checkpoint(saved_data, tmp_path):
    config = tmp_path / 'experiment.yaml'
    save_yaml(config, saved_data)
    jobs = Jobs(tmp_path)
    record = jobs.start({'kind': 'task', 'task': 'omp', 'action': 'run', 'config': str(config)})
    try:
        jobs.control(record['id'], 'pause')
        wait_status(jobs, {'paused'})
        jobs.control(record['id'], 'stop')
        stopped = wait_status(jobs, {'stopped', 'interrupted'})
        assert stopped['status'] == 'stopped'
        from cgap_sbl_jadce.recovery import recovery_path
        assert recovery_path(config, 'omp').exists()
        resumed = jobs.resume(record['id'])
        assert resumed['id'] != record['id']
        finished = wait_status(jobs, {'completed', 'failed'})
        assert finished['status'] == 'completed', jobs.log(resumed['id'])[0]
        jobs.children[resumed['id']].wait(timeout=10)
    finally:
        if jobs.active():
            jobs.control(record['id'], 'stop')
        jobs.children[record['id']].wait(timeout=10)


def test_preview_exports_read_saved_data_without_running_estimators(saved_data, tmp_path, monkeypatch):
    from cgap_sbl_jadce.experiments import estimators
    monkeypatch.setattr(estimators, 'run_estimator_stage', lambda *a, **k: pytest.fail('Preview executed an estimator'))
    config = tmp_path / 'experiment.yaml'
    save_yaml(config, saved_data)
    request = {'config': str(config), 'category': 'communication', 'split': 'evaluation', 'index': 0, 'condition': [7, 8]}
    archive = task_dir(saved_data['run']['output_dir'], 'communication') / 'evaluation' / '0000' / 'observations.npz'
    before = archive.read_bytes()
    destination = tmp_path / 'preview'
    paths = render_preview(request, destination)
    assert len(paths) == 14
    assert all(path.is_relative_to(destination) and path.exists() for path in paths)
    assert {path.suffix for path in paths} == {'.png', '.pdf'}
    assert archive.read_bytes() == before
    assert not (task_dir(saved_data['run']['output_dir'], 'communication') / 'figures').exists()
    request['condition'] = [100, 99]
    with pytest.raises(ValueError, match='saved dataset'):
        render_preview(request, destination)


def test_saved_activity_sample_options_and_exports_are_independent_of_yaml_counts(saved_data, tmp_path):
    summary_path = result_path(saved_data['run']['output_dir'], 'activity', 'summary.json')
    summary = json.loads(summary_path.read_text())
    summary.update(tuning={'samples': 2}, evaluation={'samples': 2})
    save_json(summary_path, summary)
    saved_data['run']['num_evaluation_samples'] = 999
    assert sample_options(saved_data, 'activity', 'evaluation') == (2, [])
    config = tmp_path / 'experiment.yaml'
    save_yaml(config, saved_data)
    paths = render_preview({'config': str(config), 'category': 'activity', 'split': 'evaluation', 'index': 0}, tmp_path / 'preview')
    assert [path.name for path in paths] == ['activity_realization_evaluation_0000_probability_field.png', 'activity_realization_evaluation_0000_probability_field.pdf']


def test_dashboard_navigation_does_not_launch_work(tmp_path, monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    from cgap_sbl_jadce.config import load_config
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path / 'results')
    config = tmp_path / 'experiment.yaml'
    save_yaml(config, cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    assert not app.exception
    app.button(key='open_algorithm_glc').click().run()
    assert not app.exception
    assert app.button(key='open_glc_tune')
    app.button(key='open_glc_run').click().run()
    assert not app.exception
    assert app.button(key='task_action').disabled
    assert next(widget for widget in app.selectbox if widget.label == 'Figure type').options == ['GLC performance and gamma distribution']
    app.sidebar.button[0].click().run()
    app.button(key='open_convergence_run').click().run()
    assert not app.exception
    algorithms = next(widget for widget in app.multiselect if widget.label == 'Algorithms')
    assert algorithms.disabled and not algorithms.options
    assert next(button for button in app.button if button.label == 'Regenerate publication exports').disabled
    assert missing_preview_inputs(cfg, 'convergence')
    assert not Jobs(tmp_path).records()


def test_gui_creates_setup_from_defaults_without_starting_a_task(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from cgap_sbl_jadce.config import load_config
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path / 'existing')
    source = tmp_path / 'default.yaml'
    save_yaml(source, cfg)
    original = source.read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(source))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    next(button for button in app.sidebar.button if button.label == 'Experiment setups').click().run()
    next(widget for widget in app.radio if widget.label == 'Setup action').set_value('Create setup').run()
    next(widget for widget in app.text_input if widget.label == 'Setup name').set_value('New environment')
    next(widget for widget in app.number_input if widget.label == 'MTDs N').set_value(500)
    next(widget for widget in app.radio if widget.label == 'SBL tuning policy').set_value('per_condition')
    next(button for button in app.button if button.label == 'Save setup').click().run()
    assert not app.exception
    config = tmp_path / 'results/new-environment/config.yaml'
    assert config.exists()
    result = load_config(config)
    assert result['system']['N'] == 500
    assert result['algorithms']['sbl']['tuning']['selection_mode'] == 'per_condition'
    assert result['evaluation'].get('condition_mode', 'sweeps') == 'sweeps'
    assert 'conditions' not in result['evaluation']
    assert source.read_bytes() == original
    assert not Jobs(tmp_path).records()
    assert list(config.parent.iterdir()) == [config]


def test_pause_run_another_task_then_resume(saved_data, tmp_path):
    config = tmp_path / 'experiment.yaml'
    save_yaml(config, saved_data)
    jobs = Jobs(tmp_path)
    first = jobs.start({'kind':'task','task':'omp','action':'run','config':str(config)})
    second = None
    try:
        jobs.control(first['id'],'pause')
        wait_status(jobs, {'paused'})
        assert jobs.busy() is None
        second = jobs.start({'kind':'task','task':'cosamp','action':'run','config':str(config)})
        with pytest.raises(ValueError,match='Pause the running task'):
            jobs.resume(first['id'])
        finished = wait_status(jobs, {'completed','failed'})
        assert finished['status'] == 'completed', jobs.log(second['id'])[0]
        jobs.resume(first['id'])
        deadline = time.monotonic()+30
        while time.monotonic() < deadline:
            state = jobs.task_job(config,'omp')
            if state['status'] not in {'paused','queued','running'}:
                break
            time.sleep(.05)
        assert state['status'] == 'completed', jobs.log(first['id'])[0]
        assert result_path(saved_data['run']['output_dir'],'cosamp','runtime_summary.csv').exists()
    finally:
        for job in jobs.records():
            if job['status'] in {'paused','queued','running'}:
                jobs.control(job['id'],'stop')
        for child in jobs.children.values():
            child.wait(timeout=10)


def test_completed_results_unlock_only_eligible_analysis_widgets(saved_data, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from cgap_sbl_jadce.experiments import estimators
    from cgap_sbl_jadce.gui.workflow import numerical_state, combined
    estimators.run_estimator_stage(saved_data,'omp',show_progress=False)
    estimators.tune_estimator_stage(saved_data,'sbl',show_progress=False)
    estimators.run_estimator_stage(saved_data,'sbl',show_progress=False)
    assert combined([numerical_state(saved_data,'sbl',action) for action in ('tune','run')]).status == 'completed'
    config = tmp_path / 'experiment.yaml'
    save_yaml(config,saved_data)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG',str(config))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py',default_timeout=30).run()
    assert not app.exception
    assert 'Completed' in app.button(key='open_algorithm_sbl').label
    app.button(key='open_comparison_run').click().run()
    selector = next(widget for widget in app.multiselect if widget.label == 'Algorithms')
    assert selector.options == ['MMV-OMP','SBL'] and not selector.disabled
    selector.set_value(['omp']).run()
    assert not app.button(key='task_action').disabled
    next(button for button in app.sidebar.button if button.label == 'Workflow overview').click().run()
    app.button(key='open_convergence_run').click().run()
    assert not app.exception
    selector = next(widget for widget in app.multiselect if widget.label == 'Algorithms')
    assert selector.options == ['SBL'] and not selector.disabled
    assert not app.button(key='task_action').disabled
    assert not Jobs(tmp_path).records()


def test_back_button_returns_from_every_page_to_its_parent(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from cgap_sbl_jadce.config import load_config
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path / 'results')
    config = tmp_path / 'experiment.yaml'
    save_yaml(config, cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    assert not app.exception
    assert not [button for button in app.button if button.label == 'Back']
    for method in ('sbl', 'glc'):
        app.button(key=f'open_algorithm_{method}').click().run()
        for action in ('tune', 'run'):
            app.button(key=f'open_{method}_{action}').click().run()
            assert not app.exception
            assert len([button for button in app.button if button.label == 'Back']) == 1
            app.button(key='go_back').click().run()
            assert app.session_state['page'] == 'algorithm'
        app.button(key='go_back').click().run()
        assert app.session_state['page'] == 'overview'
    app.button(key='open_activity_model').click().run()
    for task in ('kappa_calibration', 'activity'):
        app.button(key=f'open_{task}_run').click().run()
        assert not app.exception
        app.button(key='go_back').click().run()
        assert app.session_state['page'] == 'activity_model'
    app.button(key='go_back').click().run()
    for key in ('open_communication_run', 'open_spatial_dependence_run',
                'open_algorithm_omp', 'open_algorithm_cosamp',
                'open_comparison_run', 'open_convergence_run'):
        app.button(key=key).click().run()
        assert not app.exception
        app.button(key='go_back').click().run()
        assert app.session_state['page'] == 'overview'
    next(button for button in app.sidebar.button if button.label == 'Experiment setups').click().run()
    app.button(key='go_back').click().run()
    assert app.session_state['page'] == 'overview'
    assert not app.exception
    assert not Jobs(tmp_path).records()


def test_failed_worker_can_resume_with_same_request(tmp_path):
    config = tmp_path / 'experiment.yaml'
    config.write_text('{}')
    save_yaml(config, {'run': {'output_dir': str(tmp_path / 'results')}})
    jobs = Jobs(tmp_path)

    record = jobs.start({'kind': 'invalid', 'config': str(config)})
    try:
        failed = wait_status(jobs, {'failed'})
        assert failed['error'] == 'Unknown job type'
        jobs.children[record['id']].wait(timeout=10)
        resumed = jobs.resume(record['id'])
        assert resumed['id'] != record['id']
        assert resumed['request'] == record['request']
        assert wait_status(jobs, {'failed'})['error'] == 'Unknown job type'
    finally:
        for job in jobs.records():
            if job['status'] in {'paused', 'queued', 'running'}:
                jobs.control(job['id'], 'stop')
        for child in jobs.children.values():
            child.wait(timeout=10)


@pytest.mark.parametrize('status,command,enabled', [
    ('paused', 'pause', True),
    ('failed', 'run', True),
    ('stopped', 'stop', True),
    ('interrupted', 'run', True),
    ('running', 'run', False),
    ('queued', 'run', False),
    ('completed', 'run', False),
    ('paused', 'run', False),
    ('paused', 'stop', False),
])
def test_task_controls_offer_one_resume_for_live_and_recoverable_work(tmp_path, monkeypatch, status, command, enabled):
    from streamlit.testing.v1 import AppTest
    jobs = Jobs(tmp_path)
    config = tmp_path / 'config.yaml'
    save_yaml(config, {'run': {'output_dir': str(tmp_path / 'results')}})
    folder = jobs.folder / 'example'
    folder.mkdir()
    save_json(folder / 'job.json', {
        'id': 'example', 'status': status, 'created': time.time(),
        'pid': os.getpid(), 'process_token': process_token(os.getpid()),
        'request': {'kind': 'task', 'config': str(config), 'task': 'omp', 'action': 'run'},
    })
    save_json(folder / 'control.json', {'command': command})
    resumed = []
    monkeypatch.setattr(Jobs, 'resume', lambda self, job_id: resumed.append(job_id))

    monkeypatch.setattr('cgap_sbl_jadce.recovery.validate_recovery', lambda request: {})
    source = f"""
import streamlit as st
from cgap_sbl_jadce.gui.jobs import Jobs
from cgap_sbl_jadce.gui.task_page import worker_panel
worker_panel(st, Jobs({str(tmp_path)!r}), {str(config)!r}, 'omp', 'run')
"""
    app = AppTest.from_string(source).run()
    assert not app.exception
    if status == 'completed':
        assert not app.button
        assert not any(item.value == 'Task progress' for item in app.subheader)
        return
    assert [button.label for button in app.button] == ['Pause', 'Resume', 'Stop']
    resume = next(button for button in app.button if button.label == 'Resume')
    assert resume.disabled == (not enabled)
    if enabled:
        resume.click().run()
        assert not app.exception
        assert resumed == ['example']


def test_empty_dashboard_discards_deleted_setup_selection(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path / 'results/experiment')
    config = tmp_path / 'default.yaml'
    save_yaml(config, cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30)
    app.session_state['setup_path'] = tmp_path / 'results/deleted/config.yaml'
    app.run()
    assert not app.exception
    assert app.sidebar.selectbox[0].value == config
    assert not Path(cfg['run']['output_dir']).exists()
    assert not list((tmp_path / '.gui').glob('*/job.json'))
    captions = [item.value for item in app.caption]
    assert not any('Pause at a checkpoint' in text or 'Prepare your data' in text for text in captions)
    app.button(key='open_algorithm_omp').click().run()
    assert not app.exception
    assert app.button(key='task_action').disabled
    assert any(info.value == 'No results yet.' for info in app.info)
    assert not Path(cfg['run']['output_dir']).exists()
