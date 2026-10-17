from copy import deepcopy
import inspect
import json
from pathlib import Path

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.paths import result_path, task_dir, tuning_path

PREVIEWS = {
    'activity': 'Activity realization',
    'active_counts': 'Active MTD count distribution',
    'kappa': 'Kappa calibration',
    'communication': 'Communication diagnostics',
    'spatial': 'Spatial dependence matrix',
    'sbl_tune': 'SBL threshold scores',
    'gl_tune': 'GL Optuna tuning',
    'gl_tuning_sweeps': 'GL selected parameter sweeps',
    'gl_run': 'GL performance and gamma distribution',
    'glc_tune': 'GLC Optuna tuning',
    'glc_tuning_sweeps': 'GLC selected parameter sweeps',
    'sbl_tuning_sweeps': 'SBL selected threshold sweeps',
    'glc_grid_study': 'GLC paper grid study',
    'omp_run': 'OMP performance',
    'cosamp_run': 'CoSaMP performance',
    'sbl_run': 'SBL performance and gamma distribution',
    'glc_run': 'GLC performance and gamma distribution',
    'comparison': 'Algorithm comparison',
    'convergence': 'Convergence',
}
from cgap_sbl_jadce.variants import SPATIAL_METHODS
for method in SPATIAL_METHODS:
    PREVIEWS.update({method + '_tune': method.upper() + ' Optuna tuning',
                     method + '_tuning_sweeps': method.upper() + ' selected parameter sweeps',
                     method + '_run': method.upper() + ' performance and gamma distribution'})
SAMPLE_PREVIEWS = {'activity', 'communication', 'spatial'}


def main_realization_index(cfg, split, count=None):
    figures = cfg.get('figures', {})
    indices = figures.get(f'{split}_indices', [])
    fallback = indices[0] if isinstance(indices, list) and indices else 0
    index = figures.get(f'{split}_main_index', fallback)
    return min(max(index, 0), count - 1) if count is not None else index


def sample_options(cfg, category, split):
    if split not in {'tuning', 'evaluation'}:
        raise ValueError('split must be tuning or evaluation')
    group = {'communication': 'communication', 'spatial': 'spatial_dependence'}.get(category, 'activity')
    summary = json.loads(result_path(cfg['run']['output_dir'], group, 'summary.json').read_text())
    info = summary[split]
    return info['samples'], [tuple(pair) for pair in info.get('definition', {}).get('conditions', [])]


def missing_preview_inputs(cfg, category):
    if category.endswith('_tuning_sweeps'):
        from cgap_sbl_jadce.figures.estimator_parameter_sweeps import parameter_sweeps
        method = category.removesuffix('_tuning_sweeps')
        return [] if parameter_sweeps(cfg, method) else [task_dir(cfg['run']['output_dir'], method) / 'tuning']
    if category == 'glc_grid_study':
        path = result_path(cfg['run']['output_dir'], category, 'summary.json')
        if not path.exists() or json.loads(path.read_text()).get('status') != 'complete':
            return [path]
    if category.endswith('_tune'):
        method = category.removesuffix('_tune')
        names = ['thresholds.csv'] + (['trials.csv'] if method in SPATIAL_METHODS else [])
        return [tuning_path(cfg, method, name) for name in names if not tuning_path(cfg, method, name).exists()]
    sources = {
        'active_counts': [('activity', 'tuning_activity.npz'), ('activity', 'evaluation_activity.npz')],
        'kappa': [('kappa_calibration', 'kappa_sweep.csv')],
        'sbl_tune': [('sbl', 'tuning/thresholds.csv')],
        'glc_grid_study': [('glc_grid_study', 'alpha_beta.csv')],
        'comparison': [('comparison', 'aggregate.csv')],
        'convergence': [('convergence', 'convergence.npz')],
    }
    if category.endswith('_run'):
        sources[category] = [(category.removesuffix('_run'), 'aggregate.csv')]
    return [result_path(cfg['run']['output_dir'], group, name)
            for group, name in sources.get(category, [])
            if not result_path(cfg['run']['output_dir'], group, name).exists()]


def render_preview(request, destination):
    from cgap_sbl_jadce.figures.common import figure_destination
    from cgap_sbl_jadce.figures.activity_realizations import plot_activity_realization
    from cgap_sbl_jadce.figures.active_mtd_count_distribution import plot_activity_s_distribution
    from cgap_sbl_jadce.figures.kappa_calibration import plot_activity_calibration
    from cgap_sbl_jadce.figures.communication_diagnostics import plot_communication_preview
    from cgap_sbl_jadce.figures.spatial_dependence import plot_spatial_dependence
    from cgap_sbl_jadce.tasks import task_actions

    cfg = load_config(request['config'])
    category = request['category']
    if category.endswith('_tune'):
        cfg = tuning_preview_config(cfg, category.removesuffix('_tune'), request)
    if category not in PREVIEWS:
        raise ValueError('Unknown preview category')
    split, index = request.get('split', 'evaluation'), request.get('index', 0)
    if category in SAMPLE_PREVIEWS:
        count, conditions = sample_options(cfg, category, split)
        if type(index) is not int or not 0 <= index < count:
            raise ValueError(f'Realization index must be within 0..{count - 1}')
    with figure_destination(Path(cfg['run']['output_dir']), Path(destination)):
        if category.endswith('_tuning_sweeps'):
            from cgap_sbl_jadce.figures.estimator_parameter_sweeps import plot_tuning_parameter_sweeps
            return plot_tuning_parameter_sweeps(cfg, category.removesuffix('_tuning_sweeps'), formats='both')
        if category == 'activity':
            return plot_activity_realization(cfg, split, index, formats='both')
        if category == 'glc_grid_study':
            from cgap_sbl_jadce.figures.glc_parameter_tuning import plot_glc_grid_study
            return plot_glc_grid_study(cfg, formats='both')
        if category == 'active_counts':
            return plot_activity_s_distribution(cfg, formats='both')
        if category == 'kappa':
            return plot_activity_calibration(cfg, formats='both')
        if category == 'communication':
            condition = tuple(request['condition'])
            if condition not in conditions:
                raise ValueError('Choose an L/SNR condition present in the saved dataset')
            return plot_communication_preview(cfg, formats='both', show_progress=False,
                                              split=split, sample_index=index, condition=condition)
        if category == 'spatial':
            cfg = deepcopy(cfg)
            cfg['figures'] = {'tuning_indices': [], 'evaluation_indices': [], f'{split}_indices': [index]}
            return plot_spatial_dependence(cfg, formats='both', show_progress=False)
        if category.endswith('_run') and request.get('condition'):
            from cgap_sbl_jadce.figures.estimator_condition import plot_condition_scores
            return plot_condition_scores(cfg, category.removesuffix('_run'), tuple(request['condition']), formats='both')
        if category in {'comparison', 'convergence'}:
            task_id, action = category, 'run'
        else:
            method, action = category.split('_')
            task_id = method
        paths = []
        for render in task_actions(task_id)[action].plots:
            kwargs = {'formats': 'both'}
            if 'show_progress' in inspect.signature(render).parameters:
                kwargs['show_progress'] = False
            paths.extend(render(cfg, **kwargs))
        return paths


def stored_tuning_options(cfg, method):
    folder = task_dir(cfg['run']['output_dir'], method) / 'tuning'
    options = []
    for path in [folder / 'selected.json', *sorted(folder.glob('*/selected.json'))]:
        if not path.exists():
            continue
        selected = json.loads(path.read_text())
        condition = selected.get('definition', {}).get('condition')
        if condition:
            options.append(('pooled' if path.parent.name == 'pooled' else 'reference' if path.parent == folder else 'per_condition', int(condition[0]), float(condition[1])))
    return options


def tuning_preview_config(cfg, method, request):
    from cgap_sbl_jadce.settings import grouped_config
    from cgap_sbl_jadce.experiments.estimators import condition_config
    if request.get('tuning_layout'):
        if request['tuning_layout'] not in {'reference', 'per_condition', 'pooled'}:
            raise ValueError('Unknown saved tuning layout')
        cfg = grouped_config(cfg)
        cfg['algorithms'][method]['tuning']['selection_mode'] = request['tuning_layout']
    return condition_config(cfg, request['condition'], method) if request.get('condition') else cfg
