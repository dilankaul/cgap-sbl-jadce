from copy import deepcopy

import pytest

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.gui.setups import validate_setup
from cgap_sbl_jadce.settings import request_settings
from cgap_sbl_jadce.gui.previews import main_realization_index


def test_main_realization_overrides_export_order_and_handles_saved_count():
    cfg = {'figures': {'evaluation_indices': [294, 458], 'evaluation_main_index': 458}}
    assert main_realization_index(cfg, 'evaluation', 1000) == 458
    assert main_realization_index(cfg, 'evaluation', 100) == 99
    assert cfg['figures']['evaluation_main_index'] == 458
    assert main_realization_index({'figures': {'tuning_indices': [4, 37]}}, 'tuning', 100) == 4
    assert main_realization_index({'figures': {'evaluation_indices': 'all'}}, 'evaluation', 1000) == 0


@pytest.mark.parametrize('index', [-1, 1000, 1.5, True])
def test_main_realization_rejects_invalid_setting(index):
    cfg = load_config('configs/experiment.yaml')
    cfg['figures']['evaluation_main_index'] = index
    with pytest.raises(ValueError, match='main realization index'):
        validate_setup(cfg)


def test_display_defaults_preserve_scientific_request_settings():
    cfg = load_config('configs/experiment.yaml')
    changed = deepcopy(cfg)
    changed['figures']['evaluation_main_index'] = 294
    for task in ('kappa_calibration', 'activity', 'communication', 'spatial_dependence', 'omp', 'cosamp', 'sbl', 'glc'):
        request = {'kind': 'task', 'task': task, 'action': 'run'}
        assert request_settings(cfg, request) == request_settings(changed, request)


def test_dashboard_uses_updated_main_index_in_existing_session(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import cgap_sbl_jadce.gui.figure_panel as panel

    monkeypatch.setattr(panel, 'missing_preview_inputs', lambda *args: [])
    monkeypatch.setattr(panel, 'sample_options', lambda *args: (1000, []))
    monkeypatch.setattr(panel, 'interactive_figures', lambda *args: [])
    monkeypatch.setattr(panel, 'source_version', lambda *args: 'test')
    app = AppTest.from_string('''
import streamlit as st
from types import SimpleNamespace
from cgap_sbl_jadce.gui.figure_panel import figure_panel
main = st.session_state.get('main', 458)
cfg = {'figures': {'evaluation_indices': [294, 458], 'evaluation_main_index': main}}
jobs = SimpleNamespace(busy=lambda: None, records=lambda: [])
figure_panel(st, cfg, 'test.yaml', jobs, 'activity', 'run', SimpleNamespace(status='complete'))
''').run()
    assert not app.exception
    assert app.number_input[0].value == 458
    app.number_input[0].set_value(855).run()
    assert app.number_input[0].value == 855
    app.session_state['main'] = 294
    app.run()
    assert not app.exception
    assert app.number_input[0].value == 294


def test_communication_preview_waits_for_first_saved_condition(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import cgap_sbl_jadce.gui.figure_panel as panel

    pairs = []
    monkeypatch.setattr(panel, 'missing_preview_inputs', lambda *args: [])
    monkeypatch.setattr(panel, 'sample_options', lambda *args: (1000, list(pairs)))
    requests = []
    monkeypatch.setattr(panel, 'interactive_figures', lambda request: requests.append(request) or [])
    monkeypatch.setattr(panel, 'source_version', lambda *args: 'test')
    app = AppTest.from_string('''
import streamlit as st
from types import SimpleNamespace
from cgap_sbl_jadce.gui.figure_panel import figure_panel
cfg = {'figures': {'evaluation_main_index': 458}}
jobs = SimpleNamespace(busy=lambda: None, records=lambda: [])
figure_panel(st, cfg, 'test.yaml', jobs, 'communication', 'run', SimpleNamespace(status='partial'))
''').run()
    assert not app.exception
    assert any(info.value == 'No saved communication conditions in this split.' for info in app.info)
    assert not requests
    assert next(b for b in app.button if b.label == 'Regenerate publication exports').disabled
    assert not any(w.label == 'Saved L / SNR' for w in app.selectbox)

    pairs.append((20, 4.))
    app.run()
    assert not app.exception
    assert requests[-1]['condition'] == [20, 4.]
    assert not next(b for b in app.button if b.label == 'Regenerate publication exports').disabled

    pairs.clear()
    app.run()
    assert not app.exception
    assert next(b for b in app.button if b.label == 'Regenerate publication exports').disabled
