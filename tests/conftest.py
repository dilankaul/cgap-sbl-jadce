from dataclasses import replace
import numpy as np
import pytest
from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.dataset import save_activity_samples
from cgap_sbl_jadce.models.activity import ActivitySample
from cgap_sbl_jadce.io import save_json
from cgap_sbl_jadce.paths import result_path
from cgap_sbl_jadce.experiments import pipeline


@pytest.fixture
def saved_data(tmp_path):
    cfg = load_config('tests/fixtures/small_experiment.yaml')
    cfg['run'].update(output_dir=str(tmp_path), num_tuning_samples=2, num_evaluation_samples=2)
    cfg['system'].update(N=20, M=3, pilot_lengths=[5, 7], snr_db=[8, 12])
    cfg['algorithms']['K'] = 3
    from cgap_sbl_jadce.variants import SPATIAL_METHODS
    for method in SPATIAL_METHODS:
        cfg['algorithms'].setdefault(method, {}).setdefault('tuning', {}).update(
            trials=4, samples=2, startup_trials=2, alpha_low=0., alpha_high=2.)
    cfg['activity'].update(kappa_source='configured', kappa=3.78)
    cfg['tuning'].update(reference_pilot_length=5, alpha_values=[0, 2], beta_values=[0, .1])
    cfg['evaluation'].update(pilot_length=7, gamma_distribution_samples=2)
    a = np.zeros(20, dtype=bool); a[[1, 7, 13]] = True
    sample = ActivitySample(a, np.arange(40).reshape(20, 2), np.zeros((2, 2)), np.full(20, .15))
    evaluation = [sample, replace(sample, a=np.zeros(20, dtype=bool))]
    for name, samples in [('tuning', [sample, sample]), ('evaluation', evaluation)]:
        save_activity_samples(result_path(tmp_path, 'activity', f'{name}_activity.npz'), samples)
    from cgap_sbl_jadce.compatibility import preparation_settings
    save_json(result_path(tmp_path, 'activity', 'summary.json'), {'kappa': 3.78, 'D': 15, 'R': 40,
        'compatibility': {split: preparation_settings(cfg, 'activity', split) for split in ('tuning', 'evaluation')}})
    pipeline.communication_stage(cfg, show_progress=False)
    pipeline.spatial_dependence_stage(cfg, show_progress=False)
    return cfg


GUI_MODULES = {'test_gui', 'test_settings_forms', 'test_setting_ranges', 'test_experiment_setups',
               'test_live_condition_status', 'test_main_realization', 'test_tuning_conditions',
               'test_communication_conditions', 'test_editable_settings', 'test_glc_labels'}
INTEGRATION_MODULES = {'test_checkpoints', 'test_condition_lifecycle', 'test_run_reset',
                       'test_algorithm_actions', 'test_additional_conditions', 'test_estimator_tasks',
                       'test_communication_dataset', 'test_spatial_dependence_dataset', 'test_analysis_selection',
                       'test_optuna_tuning', 'test_tuning_lifecycle', 'test_gl_workflow', 'test_variant_workflow'}


def pytest_collection_modifyitems(items):
    for item in items:
        module = item.path.stem
        plotting = module in {'test_plotting', 'test_line_ranges', 'test_line_zoom_browser'} or any(
            word in item.name for word in ('diagnostics', 'exports', 'plot', 'figure', 'fonts'))
        if plotting:
            item.add_marker(pytest.mark.plotting)
        if module in GUI_MODULES:
            item.add_marker(pytest.mark.gui)
        if module in INTEGRATION_MODULES:
            item.add_marker(pytest.mark.integration)
        if not plotting and module not in GUI_MODULES | INTEGRATION_MODULES:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def lightweight_workflow_figures(request, monkeypatch):
    if request.node.get_closest_marker('plotting') or not (request.node.get_closest_marker('gui') or
                                                         request.node.get_closest_marker('integration')):
        return
    import sys
    import matplotlib.pyplot as plt
    from cgap_sbl_jadce import tasks
    from cgap_sbl_jadce.figures import common
    original = common._save
    def save(fig, stem, formats='pdf'):
        from pathlib import Path
        from cgap_sbl_jadce.io import atomic_path
        from cgap_sbl_jadce.execution_control import pause_point
        stem = Path(stem)
        destination = common._destination.get()
        if destination:
            source, folder = destination
            stem = folder / stem.resolve().relative_to(source)
        paths = []
        for ext in common.figure_formats(formats):
            path = Path(f'{stem}.{ext}')
            with atomic_path(path) as temporary:
                temporary.write_bytes(b'workflow artifact: ' + ext.encode())
            paths.append(path)
        plt.close(fig)
        pause_point()
        return paths
    for name, module in list(sys.modules.items()):
        if name.startswith('cgap_sbl_jadce.figures') and getattr(module, '_save', None) is original:
            monkeypatch.setattr(module, '_save', save)
