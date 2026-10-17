from pathlib import Path

from streamlit.testing.v1 import AppTest

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.io import save_yaml
from cgap_sbl_jadce.settings import grouped_config


def dashboard(saved_data, tmp_path, monkeypatch):
    config = tmp_path / 'source.yaml'; save_yaml(config, grouped_config(saved_data))
    monkeypatch.chdir(tmp_path); monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    return app, config


def test_ridge_is_editable_in_setup_and_cosamp_settings_with_explicit_save(saved_data, tmp_path, monkeypatch):
    app, config = dashboard(saved_data, tmp_path, monkeypatch)
    next(b for b in app.sidebar.button if b.label == 'Experiment setups').click().run()
    ridge = next(w for w in app.number_input if w.key.endswith('_cosamp_solver_cosamp_lambda_reg'))
    assert ridge.value == .01
    before = config.read_bytes()
    ridge.set_value(.02)
    assert config.read_bytes() == before
    next(b for b in app.button if b.label == 'Save changes').click().run()
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    assert load_config(config)['algorithms']['cosamp']['solver']['cosamp_lambda_reg'] == .02
    app.button(key='go_back').click().run()
    app.button(key='open_algorithm_cosamp').click().run()
    ridge = next(w for w in app.number_input if w.key.endswith('_cosamp_solver_cosamp_lambda_reg'))
    assert ridge.value == .02
    ridge.set_value(0.)
    next(b for b in app.button if b.label == 'Save changes').click().run()
    assert not app.exception
    assert load_config(config)['algorithms']['cosamp']['solver']['cosamp_lambda_reg'] == 0
    assert not list((tmp_path / '.gui').glob('*/job.json'))


def test_setup_requires_save_and_reset_discards_invalid_draft(saved_data, tmp_path, monkeypatch):
    app, config = dashboard(saved_data, tmp_path, monkeypatch)
    next(b for b in app.sidebar.button if b.label == 'Experiment setups').click().run()
    assert not any(w.label == 'Evaluation conditions' for w in app.radio)
    assert not any(w.label == 'Selected L/SNR pairs' for w in app.text_area)
    before = config.read_bytes()
    next(w for w in app.number_input if w.label == 'Pilot length stop').set_value(99).run()
    assert config.read_bytes() == before
    next(b for b in app.button if b.label == 'Reset').click().run()
    assert not app.exception
    assert next(w for w in app.number_input if w.label == 'Pilot length stop').value == 7
    assert config.read_bytes() == before
    next(w for w in app.number_input if w.label == 'SNR for pilot-length sweep').set_value(15.).run()
    app.button(key='go_back').click().run()
    next(b for b in app.sidebar.button if b.label == 'Experiment setups').click().run()
    assert next(w for w in app.number_input if w.label == 'SNR for pilot-length sweep').value == 12.
    next(w for w in app.number_input if w.label == 'SNR for pilot-length sweep').set_value(16.)
    next(b for b in app.button if b.label == 'Save changes').click().run()
    assert load_config(config)['evaluation']['snr_db'] == 16.
    assert not list((tmp_path / '.gui').glob('*/job.json'))


def test_task_settings_reset_on_back_without_saving(saved_data, tmp_path, monkeypatch):
    app, config = dashboard(saved_data, tmp_path, monkeypatch)
    app.button(key='open_algorithm_glc').click().run()
    app.button(key='open_glc_tune').click().run()
    original = config.read_bytes()
    next(w for w in app.number_input if w.key.endswith('_glc_solver_K')).set_value(4).run()
    assert config.read_bytes() == original
    next(b for b in app.button if b.label == 'Reset').click().run()
    assert next(w for w in app.number_input if w.key.endswith('_glc_solver_K')).value == 3
    next(w for w in app.number_input if w.key.endswith('_glc_solver_K')).set_value(5).run()
    app.button(key='go_back').click().run()
    app.button(key='open_glc_tune').click().run()
    assert not app.exception
    assert next(w for w in app.number_input if w.key.endswith('_glc_solver_K')).value == 3
    assert config.read_bytes() == original


def test_new_setup_reset_and_optional_conditions_need_explicit_add(saved_data, tmp_path, monkeypatch):
    from cgap_sbl_jadce.additional_conditions import additional_conditions
    app, config = dashboard(saved_data, tmp_path, monkeypatch)
    next(b for b in app.sidebar.button if b.label == 'Experiment setups').click().run()
    next(w for w in app.radio if w.label == 'Setup action').set_value('Create setup').run()
    next(w for w in app.text_input if w.label == 'Setup name').set_value('Unsaved')
    next(w for w in app.number_input if w.label == 'MTDs N').set_value(500).run()
    assert not (tmp_path / 'results/unsaved').exists()
    next(b for b in app.button if b.label == 'Reset').click().run()
    assert not app.exception
    assert next(w for w in app.text_input if w.label == 'Setup name').value == ''
    assert next(w for w in app.number_input if w.label == 'MTDs N').value == 20
    app.button(key='go_back').click().run()
    app.button(key='open_algorithm_omp').click().run()
    assert next(w for w in app.number_input if w.label == 'Additional pilot length L').value is None
    assert next(w for w in app.number_input if w.label == 'Additional SNR (dB)').value is None
    next(w for w in app.number_input if w.label == 'Additional pilot length L').set_value(9)
    next(w for w in app.number_input if w.label == 'Additional SNR (dB)').set_value(10.)
    assert additional_conditions(load_config(config), 'omp') == []
    assert next(w for w in app.number_input if w.label == 'Additional SNR (dB)').value == 10.
    original = config.read_bytes()
    next(b for b in app.button if b.label == 'Add').click().run()
    assert not app.exception
    assert additional_conditions(load_config(config), 'omp') == [(9, 10.)]
    assert config.read_bytes() == original
    assert next(w for w in app.radio if w.label == 'Condition scope').value == 'Selected conditions'
    assert next(w for w in app.multiselect if w.label == 'L / SNR conditions').value == [(9, 10.)]
    assert any(b.label == 'Prepare inputs' for b in app.button)
    assert not list((tmp_path / '.gui').glob('*/job.json'))


def test_tuning_and_sweep_anchors_can_be_saved_independently(saved_data, tmp_path, monkeypatch):
    app, config = dashboard(saved_data, tmp_path, monkeypatch)
    next(b for b in app.sidebar.button if b.label == 'Experiment setups').click().run()
    assert len([w for w in app.number_input if (w.key or '').endswith('_glc_tuning_reference_pilot_length')]) == 1
    next(w for w in app.number_input if (w.key or '').endswith('_glc_tuning_reference_pilot_length')).set_value(9)
    next(w for w in app.number_input if (w.key or '').endswith('_glc_tuning_reference_snr_db')).set_value(10.)
    next(w for w in app.number_input if w.label == 'Pilot length for SNR sweep').set_value(6)
    next(w for w in app.number_input if w.label == 'SNR for pilot-length sweep').set_value(14.)
    original = config.read_bytes()
    assert config.read_bytes() == original
    next(b for b in app.button if b.label == 'Save changes').click().run()
    assert not app.exception
    cfg = load_config(config)
    assert cfg['algorithms']['glc']['tuning']['reference_pilot_length'] == 9
    assert cfg['algorithms']['glc']['tuning']['reference_snr_db'] == 10.
    assert cfg['evaluation']['pilot_length'] == 6
    assert cfg['evaluation']['snr_db'] == 14.
    assert cfg['algorithms']['sbl']['tuning']['reference_pilot_length'] == 5
    assert cfg['communication']['reference']['reference_pilot_length'] == 5
    assert not list((tmp_path / '.gui').glob('*/job.json'))


def test_optional_table_options_are_shared_and_do_not_affect_completion(saved_data, tmp_path, monkeypatch):
    from cgap_sbl_jadce.additional_conditions import add_condition
    from cgap_sbl_jadce.experiments import estimators
    from cgap_sbl_jadce.gui.workflow import numerical_state
    estimators.run_estimator_stage(saved_data, 'omp', False)
    app, config = dashboard(saved_data, tmp_path, monkeypatch)
    app.button(key='open_algorithm_omp').click().run()
    assert numerical_state(load_config(config), 'omp').fraction == 1.
    next(b for b in app.button if b.label == 'Add').click().run()
    assert app.error and 'Enter both' in app.error[0].value
    next(w for w in app.number_input if w.label == 'Additional pilot length L').set_value(9)
    next(w for w in app.number_input if w.label == 'Additional SNR (dB)').set_value(10.)
    next(b for b in app.button if b.label == 'Add').click().run()
    assert not app.exception
    table = next(block for block in app.expander if block.label == 'Condition status')
    assert len([b for b in table.button if b.label == 'Delete']) == 1
    assert any(item.value == '9' for item in table.markdown)
    assert sum(item.value == 'Sweep' for item in table.markdown) == 3
    assert not any(block.label == 'Additional conditions (optional)' for block in app.expander)
    assert not any(b.label == 'Save condition' for b in app.button)
    assert numerical_state(load_config(config), 'omp').fraction == 1.
    app.button(key='go_back').click().run()
    app.button(key='open_algorithm_cosamp').click().run()
    table = next(block for block in app.expander if block.label == 'Condition status')
    assert len([b for b in table.button if b.label == 'Delete']) == 1
    assert any(item.value == '9' for item in table.markdown)
    assert numerical_state(load_config(config), 'cosamp').fraction == 0.
    next(b for b in app.button if b.label == 'Delete').click().run()
    table = next(block for block in app.expander if block.label == 'Condition status')
    assert not any(b.label == 'Delete' for b in table.button)
    assert sum(item.value == 'Sweep' for item in table.markdown) == 3
    app.button(key='go_back').click().run()
    app.button(key='open_algorithm_omp').click().run()
    table = next(block for block in app.expander if block.label == 'Condition status')
    assert not any(b.label == 'Delete' for b in table.button)
    assert sum(item.value == 'Sweep' for item in table.markdown) == 3
    assert numerical_state(load_config(config), 'omp').fraction == 1.
    assert not app.exception
    assert not list((tmp_path / '.gui').glob('*/job.json'))


def test_delete_button_removes_only_its_optional_row(saved_data, tmp_path, monkeypatch):
    from cgap_sbl_jadce.additional_conditions import add_condition, additional_conditions
    app, config = dashboard(saved_data, tmp_path, monkeypatch)
    cfg = load_config(config)
    add_condition(cfg, 'omp', (9, 10.))
    add_condition(cfg, 'cosamp', (11, 14.))
    app.button(key='open_algorithm_omp').click().run()
    table = next(block for block in app.expander if block.label == 'Condition status')
    deletes = [button for button in table.button if button.label == 'Delete']
    assert len(deletes) == 2
    assert deletes[0].key.endswith('(9, 10.0)')
    deletes[0].click().run()
    assert not app.exception
    assert additional_conditions(load_config(config)) == [(11, 14.)]
    table = next(block for block in app.expander if block.label == 'Condition status')
    deletes = [button for button in table.button if button.label == 'Delete']
    assert len(deletes) == 1 and deletes[0].key.endswith('(11, 14.0)')
    assert any(item.value == '11' for item in table.markdown)
    assert not any(item.value == '9' for item in table.markdown)
    assert next(widget for widget in app.number_input if widget.label == 'Additional pilot length L').value is None
    assert not any(button.label == 'Save condition' for button in app.button)
