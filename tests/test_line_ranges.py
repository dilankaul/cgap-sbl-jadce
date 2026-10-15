import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cgap_sbl_jadce.figures import algorithm_performance, estimator_threshold_scores, kappa_calibration
from cgap_sbl_jadce.gui.charts import _line_style, interactive_figures
from cgap_sbl_jadce.io import save_json, save_yaml
from cgap_sbl_jadce.paths import result_path, tuning_path


def _capture(monkeypatch, module):
    figures = []

    def save(fig, stem, formats):
        figures.append(fig)
        plt.close(fig)
        return []

    monkeypatch.setattr(module, '_save', save)
    return figures


def test_incomplete_performance_sweeps_keep_full_configured_range(saved_data, tmp_path, monkeypatch):
    cfg = saved_data
    cfg['evaluation'].update(pilot_length_start=5, pilot_length_stop=25, pilot_length_step=5,
                             snr_start=4., snr_stop=24., snr_step=4.)
    path = result_path(cfg['run']['output_dir'], 'omp', 'aggregate.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        dict(algorithm='OMP', sweep=sweep, L=L, snr_db=snr, f1=.8, nmse=.01, runtime_s=.1)
        for sweep, L, snr in [('snr_sweep', 7, 8), ('snr_sweep', 7, 12),
                              ('pilot_sweep', 10, 8), ('pilot_sweep', 15, 8)]
    ]).to_csv(path, index=False)
    captured = _capture(monkeypatch, algorithm_performance)
    algorithm_performance.make_performance_figures(cfg, method='omp', show_progress=False)
    assert len(captured) == 6
    for fig in captured:
        ax = fig.axes[0]
        expected = (4, 24) if 'SNR' in ax.get_xlabel() else (5, 25)
        assert ax.get_xlim() == expected
        if 'NMSE' in ax.get_ylabel():
            assert ax.get_yscale() == 'log'
    config = tmp_path / 'input.yaml'
    save_yaml(config, cfg)
    figures = interactive_figures({'config': str(config), 'category': 'omp_run'})
    assert len(figures) == 6
    for _, fig in figures:
        expected = (4, 24) if fig.layout.xaxis.title.text == 'SNR (dB)' else (5, 25)
        assert fig.layout.xaxis.range == expected
        assert fig.layout.xaxis.minallowed is None and fig.layout.xaxis.maxallowed is None
        assert fig.layout.yaxis.range is None
        assert fig.layout.yaxis.minallowed is None and fig.layout.yaxis.maxallowed is None


def test_threshold_and_kappa_exports_use_actual_sweep_endpoints(saved_data, tmp_path, monkeypatch):
    cfg = saved_data
    thresholds = pd.DataFrame(dict(tau=[0., .1, .2], f1=[.3, .9, .5],
                                   recall=[1., .9, .5], precision=[.2, .9, 1.]))
    path = tuning_path(cfg, 'sbl', 'thresholds.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds.to_csv(path, index=False)
    captured = _capture(monkeypatch, estimator_threshold_scores)
    estimator_threshold_scores.plot_threshold_tuning(cfg, 'sbl')
    assert captured[0].axes[0].get_xlim() == (0., .2)
    assert captured[0].axes[0].get_ylim() == (-.02, 1.02)
    path = result_path(cfg['run']['output_dir'], 'kappa_calibration', 'kappa_sweep.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(kappa=[.5, 3.78, 10.], mean_expected_active=[1., 10., 22.],
                      mean_realized_active=[1., 10., 22.])).to_csv(path, index=False)
    save_json(path.parent / 'summary.json', dict(target_mean_active=10., D=15., closest_grid_kappa=3.78))
    captured = _capture(monkeypatch, kappa_calibration)
    kappa_calibration.plot_activity_calibration(cfg)
    assert captured[0].axes[0].get_xlim() == (.5, 10.)
    config = tmp_path / 'input.yaml'
    save_yaml(config, cfg)
    for category, expected in [('sbl_tune', (0., .2)), ('kappa', (.5, 10.))]:
        fig = interactive_figures({'config': str(config), 'category': category})[-1][1]
        assert fig.layout.xaxis.range == expected
        assert fig.layout.yaxis.range is None


@pytest.mark.parametrize('log', [False, True])
def test_line_zoom_preserves_y_autorange_and_refreshes_on_data_change(log):
    import plotly.graph_objects as go
    fig = _line_style(go.Figure(go.Scatter(x=[1, 2, 3], y=[.001, .01, .1])), 'Test', 'X', 'Y', log=log)
    assert fig.layout.xaxis.range == (1., 3.)
    assert fig.layout.yaxis.range is None
    assert fig.layout.yaxis.fixedrange is None
    assert fig.layout.yaxis.type == ('log' if log else None)
    revision = fig.layout.uirevision
    changed = _line_style(go.Figure(go.Scatter(x=[1, 2, 3], y=[.01, .1, 1.])), 'Test', 'X', 'Y', log=log)
    assert changed.layout.uirevision != revision
    assert fig.layout.xaxis.minallowed is None and fig.layout.xaxis.maxallowed is None
    assert fig.layout.yaxis.minallowed is None and fig.layout.yaxis.maxallowed is None


def test_single_point_line_has_a_finite_zoom_window():
    import plotly.graph_objects as go
    fig = _line_style(go.Figure(go.Scatter(x=[5], y=[1.])), 'Test', 'X', 'Y')
    assert fig.layout.xaxis.range == (4.5, 5.5)
    assert np.isfinite(fig.layout.xaxis.range).all()
