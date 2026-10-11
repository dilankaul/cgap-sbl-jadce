from dataclasses import replace
from pathlib import Path
import runpy

import pytest

from cgap_sbl_jadce import task_cli
from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.io import save_yaml


@pytest.mark.parametrize('task_number', range(1, 10))
def test_tasks_run_stage_then_save_png(tmp_path, monkeypatch, task_number):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    path = tmp_path / 'input.yaml'
    save_yaml(path, cfg)
    original = path.read_bytes()
    called = []

    def stage(*args, **kwargs):
        called.append('stage')

    def render(cfg, **kwargs):
        called.append(kwargs)
        return [tmp_path / 'figure.png']

    monkeypatch.setitem(task_cli.TASKS, task_number,
        replace(task_cli.TASKS[task_number], stage=stage, plots=(render,)))
    assert task_cli.run_task(task_number, ['--config', str(path)]) == [tmp_path / 'figure.png']
    assert called == ['stage', {'formats': 'png'}]
    assert path.read_bytes() == original


@pytest.mark.parametrize('task_number', range(1, 10))
def test_tasks_accept_only_config_and_progress_options(task_number):
    parser = task_cli.task_parser(task_number)
    options = {option for action in parser._actions for option in action.option_strings}
    assert options == {'-h', '--help', '--config', '--no-progress'}


def test_calibration_uses_yaml_without_overrides(tmp_path, monkeypatch):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    path = tmp_path / 'input.yaml'
    save_yaml(path, cfg)
    captured = {}

    def stage(effective, **kwargs):
        captured.update(effective)
        assert kwargs == {'show_progress': False}

    monkeypatch.setitem(task_cli.TASKS, 1, replace(task_cli.TASKS[1], stage=stage, plots=()))
    task_cli.run_task(1, ['--config', str(path), '--no-progress'])
    assert captured['activity'] == cfg['activity']
    assert captured['system'] == cfg['system']
    assert captured['seeds'] == cfg['seeds']
    assert captured['task_runtime'] == {'input_config': str(path), 'no_progress': True}
    assert load_config(path) == cfg


def test_task_wrappers_are_import_safe(monkeypatch):
    monkeypatch.setattr(task_cli, 'run_task', lambda *args: pytest.fail('Import ran a task'))
    for script in Path('scripts').rglob('*.py'):
        runpy.run_path(str(script), run_name='import_check')


def test_activity_task_exports_both_splits_automatically(tmp_path, monkeypatch):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    path = tmp_path / 'input.yaml'
    save_yaml(path, cfg)
    calls = []
    monkeypatch.setattr(task_cli, 'plot_activity_realizations',
        lambda cfg, **kwargs: calls.append(kwargs) or [])
    render = task_cli.plot_activity_realizations
    monkeypatch.setitem(task_cli.TASKS, 2, replace(task_cli.TASKS[2],
        stage=lambda *args, **kw: calls.append('stage'), plots=(render,)))
    task_cli.run_task(2, ['--config', str(path)])
    assert calls[0] == 'stage'
    assert 'splits' not in calls[1]
    assert calls[1]['formats'] == 'png'


def test_activity_report_lists_split_folders_instead_of_1100_files(tmp_path, monkeypatch, capsys):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    config = tmp_path / 'input.yaml'
    save_yaml(config, cfg)
    figures = tmp_path / '02_activity_realizations' / 'figures'
    paths = [figures / 'tuning' / f'{i:04d}.png' for i in range(100)]
    paths += [figures / 'evaluation' / f'{i:04d}.png' for i in range(1000)]
    paths += [figures / 'randomized_evaluation' / f'{i:04d}.png' for i in range(1000)]
    histogram = figures / '02_activity_realized_s_distribution.png'
    paths.append(histogram)
    monkeypatch.setitem(task_cli.TASKS, 2, replace(task_cli.TASKS[2],
        stage=lambda *args, **kwargs: None,
        plots=(lambda *args, **kwargs: paths,)))
    assert task_cli.run_task(2, ['--config', str(config)]) == paths
    assert capsys.readouterr().out.splitlines() == [
        'Task 02 figures:',
        f'  - {figures / "tuning"}',
        f'  - {figures / "evaluation"}',
        f'  - {figures / "randomized_evaluation"}',
        f'  - {histogram}',
    ]
