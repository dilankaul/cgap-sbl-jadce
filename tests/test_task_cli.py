from dataclasses import replace
from pathlib import Path
import runpy

import pytest

from cgap_sbl_jadce import task_cli
from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.io import save_yaml


@pytest.mark.parametrize('task_number', sorted(task_cli.TASKS))
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


@pytest.mark.parametrize('task_number', sorted(task_cli.TASKS))
def test_tasks_accept_expected_options(task_number):
    parser = task_cli.task_parser(task_number)
    options = {option for action in parser._actions for option in action.option_strings}
    expected = {'-h', '--help', '--config', '--no-progress'}
    if task_number == 5:
        expected.add('--setups')
    assert options == expected


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


@pytest.mark.parametrize('task_number', [2, 3])
def test_figure_report_lists_one_folder_with_compact_counts(tmp_path, monkeypatch, capsys, task_number):
    from cgap_sbl_jadce.paths import task_dir
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path)
    config = tmp_path / 'input.yaml'
    save_yaml(config, cfg)
    figures = task_dir(tmp_path, task_number) / 'figures'
    paths = [figures / f'tuning_{i:04d}' / 'figure.png' for i in range(100)]
    paths += [figures / f'evaluation_{i:04d}' / 'correlated.png' for i in range(1000)]
    paths += [figures / f'evaluation_{i:04d}' / 'randomized.png' for i in range(1000)]
    monkeypatch.setitem(task_cli.TASKS, task_number, replace(task_cli.TASKS[task_number],
        stage=lambda *args, **kwargs: None,
        plots=(lambda *args, **kwargs: paths,)))
    assert task_cli.run_task(task_number, ['--config', str(config)]) == paths
    assert capsys.readouterr().out.splitlines() == [
        f'Task {task_number:02d} figures: 2100 files (100 tuning realizations, 1000 evaluation realizations)',
        f'  - {figures}',
    ]


@pytest.mark.parametrize('setups', [[(25, 8)], [(25, 8), (30, 12)]])
def test_task05_forwards_requested_setups_to_stage_and_figures(tmp_path, monkeypatch, setups):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    path = tmp_path / 'input.yaml'
    save_yaml(path, cfg)
    calls = []
    def stage(cfg, **kwargs):
        calls.append(('stage', kwargs))
    def render(cfg, **kwargs):
        calls.append(('figure', kwargs))
        return []
    monkeypatch.setitem(task_cli.TASKS, 5, replace(task_cli.TASKS[5], stage=stage, plots=(render, render)))
    task_cli.run_task(5, ['--config', str(path), '--no-progress', '--setups',
                         *[f'{L}:{snr}' for L, snr in setups]])
    assert calls == [('stage', {'show_progress': False, 'setups': setups}),
                     ('figure', {'formats': 'png', 'setups': setups}),
                     ('figure', {'formats': 'png', 'setups': setups})]


@pytest.mark.parametrize('value', ['25', '25:8:12', '0:8', '25.5:8', '25:nan'])
def test_task05_rejects_invalid_setup_syntax(value):
    with pytest.raises(SystemExit):
        task_cli.task_parser(5).parse_args(['--setups', value])
