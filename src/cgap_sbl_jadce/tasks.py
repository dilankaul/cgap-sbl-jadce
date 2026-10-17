from __future__ import annotations

from functools import partial
from dataclasses import dataclass
from collections.abc import Callable

from cgap_sbl_jadce.experiments import pipeline, estimators
from cgap_sbl_jadce.figures.kappa_calibration import plot_activity_calibration
from cgap_sbl_jadce.figures.activity_realizations import plot_activity_realizations
from cgap_sbl_jadce.figures.active_mtd_count_distribution import plot_activity_s_distribution
from cgap_sbl_jadce.figures.communication_diagnostics import plot_communication_preview
from cgap_sbl_jadce.figures.spatial_dependence import plot_spatial_dependence
from cgap_sbl_jadce.figures.anc_parameter_tuning import plot_anc_grid_study
from cgap_sbl_jadce.figures.anc_optuna_tuning import plot_anc_optuna_tuning
from cgap_sbl_jadce.figures.gl_optuna_tuning import plot_gl_optuna_tuning
from cgap_sbl_jadce.figures.estimator_parameter_sweeps import plot_tuning_parameter_sweeps
from cgap_sbl_jadce.experiments.anc_grid_study import anc_grid_study_stage
from cgap_sbl_jadce.figures.estimator_threshold_scores import plot_threshold_tuning
from cgap_sbl_jadce.figures.estimator_gamma_distribution import plot_gamma_distribution
from cgap_sbl_jadce.figures.algorithm_convergence import plot_convergence
from cgap_sbl_jadce.figures.algorithm_performance import make_performance_figures
from cgap_sbl_jadce.paths import task_dir
from cgap_sbl_jadce.checkpoints import checkpointed, is_resuming


@dataclass(frozen=True)
class Task:
    description: str
    stage: Callable
    plots: tuple[Callable, ...]
    progress: bool = False


TASKS = {
    'kappa_calibration': Task("calibrate kappa", pipeline.calibrate_kappa_stage, (plot_activity_calibration,), True),
    'activity': Task("generate activity realizations", pipeline.generate_activity_stage,
            (plot_activity_realizations, plot_activity_s_distribution), True),
    'communication': Task("generate communication datasets", pipeline.communication_stage, (plot_communication_preview,), True),
    'spatial_dependence': Task("build spatial dependence", pipeline.spatial_dependence_stage, (plot_spatial_dependence,), True),
    'omp': Task("run OMP", partial(estimators.run_estimator_stage, method="omp"),
            (partial(make_performance_figures, method="omp"),), True),
    'cosamp': Task("run CoSaMP", partial(estimators.run_estimator_stage, method="cosamp"),
            (partial(make_performance_figures, method="cosamp"),), True),
    'sbl': Task("run SBL", partial(estimators.run_estimator_stage, method="sbl"),
            (partial(plot_gamma_distribution, method="sbl"),
             partial(make_performance_figures, method="sbl")), True),
    'anc': Task("run CGAP-SBL-ANC", partial(estimators.run_estimator_stage, method="anc"),
            (partial(plot_gamma_distribution, method="anc"), partial(make_performance_figures, method="anc")), True),
    'gl': Task('run CGAP-SBL-GL', partial(estimators.run_estimator_stage, method='gl'),
            (partial(plot_gamma_distribution, method='gl'), partial(make_performance_figures, method='gl')), True),
    'anc_grid_study': Task('study ANC parameter grid', anc_grid_study_stage, (plot_anc_grid_study,), True),
    'comparison': Task("compare algorithms", estimators.comparison_stage, (make_performance_figures,), True),
    'convergence': Task("analyze convergence", estimators.convergence_stage, (plot_convergence,), True),
}


def task_actions(task_id: str) -> dict[str, Task]:
    for method, spec in estimators.ESTIMATORS.items():
        if spec.task == task_id and spec.tuning_grid:
            plots = (partial(plot_threshold_tuning, method=method),)
            if method == 'anc':
                plots = (plot_anc_optuna_tuning,) + plots
            if method == 'gl':
                plots = (plot_gl_optuna_tuning,) + plots
            return {
                'tune': Task(f'tune {spec.name}', partial(estimators.tune_estimator_stage, method=method), plots, True),
                'run': TASKS[task_id],
            }
    return {'run': TASKS[task_id]}


@checkpointed
def run_task(task_id: str, cfg: dict, action: str = "run", show_progress: bool = True,
             conditions: list | None = None, methods: list | None = None,
             realization_index: int | None = None, condition: tuple | None = None) -> list:
    request = None
    if cfg.get('task_runtime', {}).get('input_config'):
        request = {'kind': 'task', 'config': cfg['task_runtime']['input_config'], 'task': task_id, 'action': action}
        for key, value in [('conditions', conditions), ('methods', methods),
                           ('realization_index', realization_index), ('condition', condition)]:
            if value is not None:
                request[key] = value
        if is_resuming():
            from cgap_sbl_jadce.recovery import restore_completed_outputs
            restored = restore_completed_outputs(request)
            if restored is not None:
                if action == 'tune':
                    from cgap_sbl_jadce.figures.estimator_parameter_sweeps import plot_tuning_parameter_sweeps
                    restored = list(dict.fromkeys(restored + plot_tuning_parameter_sweeps(cfg, task_id)))
                return restored
    actions = task_actions(task_id)
    if action not in actions:
        raise ValueError(f"{TASKS[task_id].description.capitalize()} has no {action} action.")
    task = actions[action]
    kwargs = {"show_progress": show_progress} if task.progress else {}
    if conditions is not None:
        if task_id not in {spec.task for spec in estimators.ESTIMATORS.values()} | {'comparison', 'communication'}:
            raise ValueError('Condition selection applies to algorithm evaluation and comparison.')
        kwargs['conditions'] = conditions
    if task_id in {'comparison', 'convergence', 'communication'} and methods is not None:
        kwargs['methods'] = methods
    if task_id == 'convergence':
        if realization_index is not None:
            kwargs['realization_index'] = realization_index
        if condition is not None:
            kwargs['condition'] = condition
    task.stage(cfg, **kwargs)
    paths = []
    owned_plots = (action == 'tune' and task.stage.func is estimators.tune_estimator_stage
                   if isinstance(task.stage, partial) else task.stage is anc_grid_study_stage)
    if owned_plots:
        import json
        from pathlib import Path
        from cgap_sbl_jadce.paths import tuning_path
        summaries = ([tuning_path(estimators.condition_config(cfg, pair, task_id), task_id, 'summary.json')
                      for pair in estimators.tuning_conditions(cfg, conditions, task_id)] if action == 'tune' else
                     [task_dir(cfg['run']['output_dir'], task_id) / 'summary.json'])
        for summary_path in summaries:
            summary = json.loads(summary_path.read_text())
            paths.extend((summary_path.parent / name).resolve() for name in summary.get('outputs', {})
                         if Path(name).suffix == '.png')
    render_configs = ([estimators.condition_config(cfg, pair, task_id) for pair in estimators.tuning_conditions(cfg, conditions, task_id)]
                      if action == 'tune' else [cfg])
    if task_id == 'communication' and conditions is not None:
        for pair in conditions:
            paths.extend(plot_communication_preview(cfg, formats='png', show_progress=show_progress, condition=tuple(pair)))
    for render_cfg in render_configs:
        for render in (() if owned_plots or task_id == 'communication' and conditions is not None else task.plots):
            kwargs = {"formats": "png"}
            if (render.func if isinstance(render, partial) else render) in (plot_activity_realizations, plot_communication_preview, plot_spatial_dependence, make_performance_figures):
                kwargs["show_progress"] = show_progress
            paths.extend(render(render_cfg, **kwargs))
    if action == 'tune':
        from cgap_sbl_jadce.figures.estimator_parameter_sweeps import sweep_figure_dir
        paths.extend(sweep_figure_dir(cfg, task_id).glob(f'{task_id}_*_vs_*.png'))
    print(f"{TASKS[task_id].description.capitalize()} figures: {len(set(paths))} files")
    if paths:
        print(f"  - {task_dir(cfg['run']['output_dir'], task_id) / 'figures'}")
    if request:
        from cgap_sbl_jadce.recovery import record_completed_outputs
        record_completed_outputs(request, paths)
    return paths
