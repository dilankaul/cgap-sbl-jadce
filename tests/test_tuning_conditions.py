from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from cgap_sbl_jadce.additional_conditions import add_condition
from cgap_sbl_jadce.checkpoints import Checkpoint
from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.experiments import estimators
from cgap_sbl_jadce.gui.tuning_conditions import tuning_plan
from cgap_sbl_jadce.io import save_yaml
from cgap_sbl_jadce.paths import tuning_path
from cgap_sbl_jadce.settings import grouped_config


@pytest.mark.parametrize('method', ['sbl', 'glc'])
def test_tune_table_tracks_conditions_checkpoints_compatibility_and_optional_pairs(saved_data, tmp_path, monkeypatch, method):
    cfg = grouped_config(saved_data)
    cfg['algorithms'][method]['tuning']['selection_mode'] = 'per_condition'
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    assert len(tuning_plan(config, method).start) == 3
    estimators.tune_estimator_stage(cfg, method, False, conditions=[(5, 12.)])
    current = estimators.condition_config(cfg, (7, 8.), method)
    checkpoint = Checkpoint(tuning_path(current, method, ''), estimators.tuning_definition(current, method))
    checkpoint.save('pair_0000_0000', {'nmse': .2, 'iterations': 1}, gamma=np.ones(cfg['system']['N']))
    add_condition(cfg, method, (9, 10.))
    folder = tuning_path(cfg, method, '').parent
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in folder.rglob('*') if path.is_file()}
    plan = tuning_plan(config, method)
    assert plan.completed == [(5, 12.)]
    assert plan.resume == [(7, 8.)]
    assert plan.start == [(7, 12.)] and not plan.blocked
    assert (9, 10.) not in plan.pairs
    assert tuning_plan(config, method, [(9, 10.)]).start == [(9, 10.)]
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items())

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    app.button(key=f'open_algorithm_{method}').click().run()
    app.button(key=f'open_{method}_tune').click().run()
    assert not app.exception
    table = next(expander for expander in app.expander if expander.label == 'Condition status')
    text = [widget.value for widget in table.markdown]
    assert '**Pilot length L**' in text and '**SNR (dB)**' in text and '**Status**' in text
    assert text.count('Completed') == 1 and text.count('Unfinished') == 1 and text.count('Not started') == 2
    assert text.count('Optional') == 2
    assert any(button.label == 'Add' for button in table.button)
    assert any(button.label == 'Delete' for button in table.button)
    assert any(widget.label == 'Condition scope' for widget in app.radio)
    assert not list((tmp_path / '.gui').glob('*/job.json'))
    next(widget for widget in app.number_input if widget.label == 'Additional pilot length L').set_value(8)
    next(widget for widget in app.number_input if widget.label == 'Additional SNR (dB)').set_value(16.)
    next(button for button in app.button if button.label == 'Add').click().run()
    assert not app.exception
    from cgap_sbl_jadce.additional_conditions import additional_conditions
    assert (8, 16.) in additional_conditions(cfg, 'cosamp')
    assert next(widget for widget in app.multiselect if widget.label == 'L / SNR conditions').value == [(8, 16.)]

    cfg['algorithms'][method]['solver']['delta'] *= 2
    save_yaml(config, cfg)
    changed = tuning_plan(config, method)
    assert set(changed.blocked) == {(5, 12.), (7, 8.)}
    app.run()
    assert not app.exception
    assert sum(widget.value == 'Incompatible' for widget in app.markdown) == 2
    cfg['algorithms'][method]['solver']['delta'] /= 2
    save_yaml(config, cfg)
    assert tuning_plan(config, method).completed == [(5, 12.)]
    assert tuning_plan(config, method).resume == [(7, 8.)]
    path = tuning_path(cfg, method, 'thresholds.csv', (5, 12.))
    path.write_text(path.read_text() + '\n')
    assert (5, 12.) in tuning_plan(config, method).blocked


def test_tuning_table_plans_unstarted_conditions_without_prepared_inputs(tmp_path):
    cfg = grouped_config(load_config('tests/fixtures/small_experiment.yaml'))
    cfg['run']['output_dir'] = str(tmp_path / 'empty')
    cfg['algorithms']['sbl']['tuning']['selection_mode'] = 'per_condition'
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    plan = tuning_plan(config, 'sbl')
    assert plan.start == estimators.tuning_conditions(cfg, method='sbl')
    assert not plan.completed and not plan.resume and not plan.blocked


def test_tuning_table_distinguishes_pending_recovery_from_changed_settings(saved_data, tmp_path):
    from cgap_sbl_jadce.recovery import begin_recovery
    cfg = grouped_config(saved_data)
    cfg['algorithms']['sbl']['tuning']['selection_mode'] = 'per_condition'
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    request = {'kind': 'task', 'config': str(config), 'task': 'sbl', 'action': 'tune',
               'conditions': [[7, 12.]]}
    begin_recovery(request)
    plan = tuning_plan(config, 'sbl')
    assert plan.resume == [(7, 12.)] and len(plan.start) == 2
    cfg['algorithms']['sbl']['solver']['delta'] *= 2
    save_yaml(config, cfg)
    assert set(tuning_plan(config, 'sbl').blocked) == {(7, 12.)}
