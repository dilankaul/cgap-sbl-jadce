import json
import shutil
from dataclasses import replace

import numpy as np
import pytest

from cgap_sbl_jadce.experiments import estimators
from cgap_sbl_jadce.experiments.readiness import blocking_reasons
from cgap_sbl_jadce.gui.task_page import analysis_options
from cgap_sbl_jadce.paths import result_path, task_dir


def test_one_completed_greedy_method_can_be_compared_but_not_converged(saved_data, monkeypatch):
    assert analysis_options(saved_data, 'comparison') == []
    estimators.run_estimator_stage(saved_data, 'omp', show_progress=False)
    assert analysis_options(saved_data, 'comparison') == ['omp']
    assert analysis_options(saved_data, 'convergence') == []
    assert not blocking_reasons(saved_data, 'comparison', methods=['omp'])
    assert blocking_reasons(saved_data, 'convergence', methods=['omp'])
    spec = estimators.ESTIMATORS['omp']
    monkeypatch.setitem(estimators.ESTIMATORS,'omp',replace(spec,solve=lambda *a,**k: pytest.fail('Comparison ran solver')))
    summary = estimators.comparison_stage(saved_data, methods=['omp'], show_progress=False)
    assert summary['algorithms'] == ['MMV-OMP']
    assert summary['metric_rows'] == 8
    with pytest.raises(ValueError, match='completed SBL'):
        estimators.convergence_stage(saved_data, methods=['omp'], show_progress=False)


def test_convergence_requires_run_not_just_tuning_and_can_use_sbl_alone(saved_data):
    estimators.tune_estimator_stage(saved_data, 'sbl', show_progress=False)
    estimators.tune_estimator_stage(saved_data, 'glc', show_progress=False)
    assert analysis_options(saved_data, 'convergence') == []
    estimators.run_estimator_stage(saved_data, 'sbl', show_progress=False)
    shutil.rmtree(task_dir(saved_data['run']['output_dir'],'spatial_dependence'))
    assert analysis_options(saved_data, 'convergence') == ['sbl']
    assert not blocking_reasons(saved_data,'convergence', methods=['sbl'])
    summary = estimators.convergence_stage(saved_data, methods=['sbl'], realization_index=0, show_progress=False)
    assert summary['methods'] == ['sbl']
    with np.load(result_path(saved_data['run']['output_dir'],'convergence','convergence.npz')) as data:
        assert 'sbl_nmse_history' in data and 'glc_nmse_history' not in data
    from cgap_sbl_jadce.figures.algorithm_convergence import plot_convergence
    from cgap_sbl_jadce.gui.charts import interactive_figures
    from cgap_sbl_jadce.io import save_yaml
    config = task_dir(saved_data['run']['output_dir'],'convergence') / 'input.yaml'
    save_yaml(config,saved_data)
    assert plot_convergence(saved_data)[0].exists()
    charts = interactive_figures({'config':str(config),'category':'convergence'})
    assert len(charts[0][1].data) == 1


def test_partial_run_is_available_only_for_its_completed_condition(saved_data):
    estimators.run_estimator_stage(saved_data,'cosamp',show_progress=False,conditions=[[7,8]])
    assert analysis_options(saved_data,'comparison',conditions=[[7,8]]) == ['cosamp']
    assert analysis_options(saved_data,'comparison',conditions=[[5,12]]) == []
    assert analysis_options(saved_data,'comparison') == []
    from cgap_sbl_jadce.paths import run_path
    path = run_path(saved_data, 'cosamp', 'summary.json', (7, 8))
    summary = json.loads(path.read_text())
    summary['definition']['revision'] = -1
    path.write_text(json.dumps(summary))
    assert analysis_options(saved_data,'comparison',conditions=[[7,8]]) == []
