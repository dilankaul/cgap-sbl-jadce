from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.io import save_json, save_yaml
from cgap_sbl_jadce.paths import task_dir, tuning_path


def test_glc_dashboard_and_settings_keep_existing_task_ids(saved_data, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from cgap_sbl_jadce.settings import grouped_config
    cfg = grouped_config(saved_data)
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    original = config.read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CGAP_GUI_CONFIG', str(config))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/cgap_sbl_jadce/gui/app.py', default_timeout=30).run()
    assert not app.exception
    assert 'CGAP-SBL-GLC' in app.button(key='open_algorithm_glc').label
    assert 'CGAP-SBL-GL\n' in app.button(key='open_algorithm_gl').label
    app.button(key='open_algorithm_glc').click().run()
    assert app.title[0].value == 'CGAP-SBL-GLC'
    assert any(item.value == 'Global–Local Contrast' for item in app.caption)
    app.button(key='open_glc_tune').click().run()
    assert not app.exception
    assert app.title[0].value == 'Tune CGAP-SBL-GLC'
    assert any(item.label == 'CGAP-SBL-GLC settings' for item in app.expander)
    assert next(item for item in app.radio if item.label == 'CGAP-SBL-GLC tuning policy')
    assert config.read_bytes() == original


def test_saved_glc_scores_show_glc_in_interactive_and_native_plots(tmp_path, monkeypatch):
    from cgap_sbl_jadce.gui.charts import interactive_figures
    from cgap_sbl_jadce.figures import algorithm_performance
    cfg = load_config('configs/experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path / 'results')
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    folder = task_dir(cfg['run']['output_dir'], 'comparison')
    folder.mkdir(parents=True)
    path = folder / 'aggregate.csv'
    rows = [{'algorithm': name, 'L': 30, 'snr_db': snr, 'sweep': 'snr_sweep',
             'f1': .9, 'nmse': .1, 'runtime_s': .01}
            for name in ('CGAP-SBL-GLC', 'CGAP-SBL-GL') for snr in (4., 8.)]
    pd.DataFrame(rows).to_csv(path, index=False)
    original = path.read_bytes()
    figures = interactive_figures({'config': str(config), 'category': 'comparison'})
    assert figures
    for _, fig in figures:
        assert {trace.name for trace in fig.data} == {'CGAP-SBL-GLC', 'CGAP-SBL-GL'}
    legends = []
    def capture(fig, *args, **kwargs):
        legends.append({text.get_text() for text in fig.axes[0].get_legend().get_texts()})
        plt.close(fig)
        return []
    monkeypatch.setattr(algorithm_performance, '_save', capture)
    algorithm_performance.make_performance_figures(cfg, show_progress=False)
    assert legends and all(names == {'CGAP-SBL-GLC', 'CGAP-SBL-GL'} for names in legends)
    assert path.read_bytes() == original


def test_saved_glc_tuning_diagnostics_use_glc_labels_without_rewriting_selection(tmp_path, monkeypatch):
    from cgap_sbl_jadce.gui.charts import interactive_figures
    from cgap_sbl_jadce.figures import spatial_optuna_tuning, estimator_threshold_scores
    cfg = load_config('configs/experiment.yaml')
    cfg['run']['output_dir'] = str(tmp_path / 'results')
    config = tmp_path / 'source.yaml'
    save_yaml(config, cfg)
    folder = tuning_path(cfg, 'glc', '')
    folder.mkdir(parents=True)
    selected = {'definition': {'method': 'CGAP-SBL-GLC', 'condition': [30, 12.]},
                'parameters': {'alpha': 2., 'beta': .05, 'tau': .1},
                'tuning_scores': {'trial': 0, 'f1': .9}}
    save_json(folder / 'selected.json', selected)
    pd.DataFrame([{'trial': 0, 'alpha': 2., 'beta': .05, 'tau': .1, 'f1': .9}]).to_csv(folder / 'trials.csv', index=False)
    pd.DataFrame([{'tau': .1, 'f1': .9, 'recall': .9, 'precision': .9}]).to_csv(folder / 'thresholds.csv', index=False)
    original = (folder / 'selected.json').read_bytes()
    figures = interactive_figures({'config': str(config), 'category': 'glc_tune'})
    assert len(figures) == 3
    assert all('GLC' in fig.layout.title.text and 'ANC' not in fig.layout.title.text for _, fig in figures)
    titles = []
    def capture(fig, *args, **kwargs):
        titles.append(fig.axes[0].get_title())
        plt.close(fig)
        return []
    for module in (spatial_optuna_tuning, estimator_threshold_scores):
        monkeypatch.setattr(module, '_save', capture)
    spatial_optuna_tuning.plot_spatial_optuna_tuning(cfg, 'glc')
    estimator_threshold_scores.plot_threshold_tuning(cfg, 'glc')
    assert len(titles) == 3 and all('GLC' in title and 'ANC' not in title for title in titles)
    assert (folder / 'selected.json').read_bytes() == original


def test_glc_reports_preserve_internal_identifiers(capsys):
    from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
    from cgap_sbl_jadce.gui.workflow import CardState
    from cgap_sbl_jadce.reporting import print_task_report
    from cgap_sbl_jadce.tasks import task_actions
    assert ESTIMATORS['glc'].name == 'CGAP-SBL-GLC'
    assert task_actions('glc')['tune'].description == 'tune CGAP-SBL-GLC'
    assert task_actions('glc')['run'].description == 'run CGAP-SBL-GLC'
    assert CardState('completed', 'CGAP-SBL-GLC').summary == 'CGAP-SBL-GLC'
    print_task_report('CGAP-SBL-GLC', {'algorithms': ['CGAP-SBL-GLC', 'CGAP-SBL-GL']})
    output = capsys.readouterr().out
    assert 'CGAP-SBL-GLC COMPLETE' in output and 'CGAP-SBL-ANC' not in output
