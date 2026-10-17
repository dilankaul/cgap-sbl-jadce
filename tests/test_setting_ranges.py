from copy import deepcopy
from pathlib import Path

import pytest

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.settings import (range_values, grouped_config, calibration_settings,
                                     evaluation_settings, tuning_settings, grid_study_settings)
from cgap_sbl_jadce.gui.setups import validate_setup
from cgap_sbl_jadce.experiments.estimators import _glc_grid, evaluation_conditions
from cgap_sbl_jadce.io import save_yaml


def test_default_sweeps_keep_all_original_values_and_inclusive_endpoints():
    cfg = load_config('configs/experiment.yaml')
    cal, tune, evaluation = calibration_settings(cfg), grid_study_settings(cfg), evaluation_settings(cfg)
    assert cal['kappa_scan'] == [i / 2 for i in range(1, 21)]
    assert cal['kappa_refinement_steps'] == [.1, .01]
    assert range_values(tune, 'alpha') == list(range(11))
    assert range_values(tune, 'beta') == [i / 100 for i in range(11)]
    assert evaluation['pilot_lengths'] == list(range(20, 46, 5))
    assert evaluation['snr_values'] == list(range(4, 25, 4))
    assert len(_glc_grid(cfg)) == 121
    assert len({value[:2] for value in evaluation_conditions(cfg)}) == 11
    validate_setup(cfg)


@pytest.mark.parametrize('name,start,stop,step,expected', [
    ('beta', 0., .03, .01, [0., .01, .02, .03]),
    ('alpha', .1, .3, .1, [.1, .2, .3]),
    ('kappa', 3.775, 3.795, .005, [3.775, 3.78, 3.785, 3.79, 3.795]),
    ('snr', -4, 4, 4, [-4., 0., 4.]),
    ('tau', .1, .1, .001, [.1]),
])
def test_decimal_ranges_include_stop_without_float_drift(name, start, stop, step, expected):
    fields = {name + '_start': start, name + '_stop': stop, name + '_step': step}
    assert range_values(fields, name) == expected


@pytest.mark.parametrize('fields', [
    {'alpha_start': 0., 'alpha_stop': 1., 'alpha_step': 0.},
    {'alpha_start': 2., 'alpha_stop': 1., 'alpha_step': .1},
    {'alpha_start': 0., 'alpha_stop': .1, 'alpha_step': .03},
    {'alpha_start': 0., 'alpha_stop': 1.},
    {'alpha_start': float('nan'), 'alpha_stop': 1., 'alpha_step': .1},
])
def test_invalid_range_is_rejected_before_tuning(fields):
    with pytest.raises(ValueError):
        range_values(fields, 'alpha')


def test_fractional_pilot_length_range_is_rejected():
    with pytest.raises(ValueError, match='integer'):
        range_values({'pilot_length_start': 10, 'pilot_length_stop': 20, 'pilot_length_step': 2.5},
                     'pilot_length', integer=True)


def test_old_explicit_grids_remain_readable_and_regular_grids_migrate_exactly(saved_data):
    cfg = grouped_config(saved_data)
    assert _glc_grid(cfg) == _glc_grid(saved_data)
    assert evaluation_conditions(cfg) == evaluation_conditions(saved_data)
    assert calibration_settings(cfg) == calibration_settings(saved_data)
    assert 'alpha_values' not in cfg['algorithms']['glc']['tuning']
    assert 'pilot_lengths' not in cfg['evaluation']

    assert cfg['kappa_calibration']['kappa_scan'] == saved_data['activity']['kappa_scan']
    original = deepcopy(saved_data)
    original['tuning']['alpha_values'] = [0., 2., 7.]
    assert _glc_grid(grouped_config(original)) == _glc_grid(original)


def test_gui_exposes_all_range_controls_and_saves_edited_grid(saved_data, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    cfg = grouped_config(saved_data)

    cfg['kappa_calibration'].pop('kappa_scan', None)
    cfg['kappa_calibration'].update(kappa_start=.5, kappa_stop=8., kappa_step=.5)
    config = tmp_path / 'settings.yaml'; save_yaml(config, cfg)
    monkeypatch.chdir(tmp_path); monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    next(b for b in app.sidebar.button if b.label == 'Experiment setups').click().run()
    labels = [widget.label for widget in app.number_input]
    for parameter in ('Alpha', 'Beta', 'Kappa', 'Pilot length', 'SNR'):
        assert all(parameter + ' ' + part in labels for part in ('start', 'stop', 'step'))
    assert not any(widget.label in {'Alpha values', 'Beta values', 'Kappa scan', 'Pilot lengths', 'Snr values'} for widget in app.text_input)
    next(w for w in app.number_input if w.label == 'Alpha stop').set_value(4.)
    assert not app.error, [error.value for error in app.error]
    assert not app.exception
    next(b for b in app.button if b.label == 'Save changes').click().run()
    assert not app.exception
    saved = load_config(config)
    assert range_values(grid_study_settings(saved), 'alpha') == [0., 2., 4.]
    assert not list((tmp_path / '.gui').glob('*/job.json'))
