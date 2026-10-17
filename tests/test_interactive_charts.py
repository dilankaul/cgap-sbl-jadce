import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('plotly')

from cgap_sbl_jadce.gui.charts import interactive_figures
from cgap_sbl_jadce.experiments import estimators, pipeline
from cgap_sbl_jadce.paths import result_path
from cgap_sbl_jadce.io import save_yaml
from cgap_sbl_jadce.figures.activity_realizations import activity_probability_grid


def request_file(cfg, tmp_path, **selection):
    config = tmp_path / 'input.yaml'
    save_yaml(config, cfg)
    return {'config': str(config), **selection}


def test_activity_view_preserves_actual_probability_field_and_zero_regions(saved_data, tmp_path, monkeypatch):
    request = request_file(saved_data, tmp_path, category='activity', split='evaluation', index=0)
    figure = interactive_figures(request)[0][1]
    _, _, expected = activity_probability_grid(np.zeros((2, 2)), 40, 15, 3.78)
    np.testing.assert_array_equal(np.asarray(figure.data[0].z), expected.filled(np.nan))
    assert np.any(np.asarray(figure.data[0].z) == 0)
    assert figure.data[0].zmin == 0 and figure.data[0].zmax == 1
    assert figure.layout.dragmode == 'pan'
    assert len(figure.data[1].customdata) == 17


def test_communication_hsv_has_exact_magnitude_phase_and_does_not_read_images(saved_data, tmp_path, monkeypatch):
    from cgap_sbl_jadce.communication_dataset import load_communication_sample
    from cgap_sbl_jadce.paths import task_dir
    request = request_file(saved_data, tmp_path, category='communication', split='evaluation', index=0, condition=[7, 8])
    data = load_communication_sample(task_dir(saved_data['run']['output_dir'], 'communication') / 'evaluation', 0)[7, 8]
    original = Path.read_bytes
    def no_images(path):
        assert path.suffix not in {'.png', '.pdf'}
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', no_images)
    figures = dict(interactive_figures(request))
    assert len(figures) == 7
    values = np.asarray(figures['H magnitude and phase'].data[0].customdata)
    np.testing.assert_allclose(values[..., 0], data.H.real)
    np.testing.assert_allclose(values[..., 1], data.H.imag)
    np.testing.assert_allclose(values[..., 2], np.abs(data.H))
    np.testing.assert_allclose(values[..., 3], np.angle(data.H))
    np.testing.assert_allclose(figures['Effective-channel row energy'].data[0].y, np.sum(np.abs(data.Z)**2, axis=1))


def test_performance_chart_uses_true_scores_and_log_nmse(saved_data, tmp_path):
    estimators.run_estimator_stage(saved_data, 'omp', False, conditions=[[7, 8]])
    request = request_file(saved_data, tmp_path, category='omp_run')
    figures = interactive_figures(request)
    assert len(figures) == 3
    nmse = next(figure for label, figure in figures if label.startswith('NMSE'))
    assert nmse.layout.yaxis.type == 'log'
    expected = pd.read_csv(result_path(saved_data['run']['output_dir'], 'omp', 'aggregate.csv'))
    np.testing.assert_allclose(nmse.data[0].y, expected.nmse)


def test_optional_condition_heatmaps_and_independent_tuning_use_saved_data(saved_data, tmp_path):
    cfg = saved_data
    cfg['tuning']['selection_mode'] = 'per_condition'
    pipeline.communication_stage(cfg, False)
    from cgap_sbl_jadce.additional_conditions import add_condition
    add_condition(cfg, 'glc', (5, 8.))
    pipeline.communication_stage(cfg, False, conditions=[[5, 8]], methods=['glc'])
    estimators.tune_estimator_stage(cfg, 'glc', False, conditions=[[5, 8]])
    estimators.run_estimator_stage(cfg, 'glc', False, conditions=[[5, 8]])
    request = request_file(cfg, tmp_path, category='glc_tune', condition=[5, 8])
    assert [label for label, _ in interactive_figures(request)] == [
        'TPE search history', 'Sampled parameter pairs', 'Threshold scores',
        'Row norm threshold scores', 'Effective-channel energy · tuning realization 0']
    request['category'] = 'glc_run'
    request.pop('condition')
    figures = interactive_figures(request)
    assert len(figures) == 3
    assert all(figure.data[0].type == 'heatmap' for label, figure in figures)
    from cgap_sbl_jadce.figures.algorithm_performance import make_performance_figures
    assert len(make_performance_figures(cfg, method='glc', show_progress=False)) == 3


def test_square_matrix_cells_preserve_complex_matrix_proportions_and_entries(saved_data, tmp_path):
    from cgap_sbl_jadce.gui.charts import _complex
    matrix = np.arange(24).reshape(3, 8) * (1 + 2j)
    figure = _complex(matrix, 'Matrix', 'Column', 'Row')
    assert figure.layout.xaxis.range is None
    assert figure.layout.yaxis.range is None
    assert figure.layout.yaxis.scaleanchor is False
    assert figure.layout.yaxis2.scaleanchor is False
    np.testing.assert_allclose(np.asarray(figure.data[0].customdata)[..., 0], matrix.real)
    np.testing.assert_allclose(np.asarray(figure.data[0].customdata)[..., 1], matrix.imag)
    request = request_file(saved_data, tmp_path, category='spatial', split='evaluation', index=0)
    figure = interactive_figures(request)[0][1]
    assert figure.layout.yaxis.scaleanchor == 'x'
    assert figure.layout.yaxis.scaleratio == 1
    request.update(category='communication', condition=[7, 8])
    correlation = dict(interactive_figures(request))['Pilot cross-correlation']
    assert correlation.layout.yaxis.scaleanchor == 'x'
    assert correlation.layout.yaxis.scaleratio == 1
