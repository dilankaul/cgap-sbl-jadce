
from cgap_sbl_jadce.config import resolve_activity_kappa
from cgap_sbl_jadce.communication_dataset import validate_communication_dataset
from cgap_sbl_jadce.spatial_dependence_dataset import validate_spatial_dependence_dataset
from cgap_sbl_jadce.experiments.estimators import ESTIMATORS, evaluation_conditions, evaluation_selections, tuning_conditions, completed_methods, convergence_methods


def blocking_reasons(cfg, task, action='run', conditions=None, methods=None, condition=None):
    reasons = []
    def check(operation):
        try:
            operation()
        except (OSError, ValueError, KeyError) as error:
            reasons.append(str(error))

    def activity():
        from cgap_sbl_jadce.compatibility import validate_preparation
        for split in ('tuning', 'evaluation'):
            validate_preparation(cfg, 'activity', split)

    if task == 'activity':
        check(lambda: resolve_activity_kappa(cfg))
    if task in {'communication', 'spatial_dependence'}:
        check(activity)
    if task in ESTIMATORS:
        spec = ESTIMATORS[task]
        split = 'tuning' if action == 'tune' else 'evaluation'
        pairs = tuning_conditions(cfg, conditions, task) if action == 'tune' else list(dict.fromkeys(v[:2] for v in evaluation_conditions(cfg, conditions, method=task)))
        if action == 'tune':
            from cgap_sbl_jadce.settings import tuning_settings, pooled_assignment
            if tuning_settings(cfg, task)['selection_mode'] == 'pooled':
                try:
                    pairs = list(dict.fromkeys((row['L'], row['snr_db']) for row in pooled_assignment(cfg)))
                except (ValueError, KeyError) as error:
                    reasons.append(str(error))
        check(lambda: validate_communication_dataset(cfg, split, pairs))
        if spec.spatial:
            check(lambda: validate_spatial_dependence_dataset(cfg, split))
        if action == 'run' and spec.tuning_grid and not reasons:
            check(lambda: evaluation_selections(cfg, task, evaluation_conditions(cfg, conditions, method=task)))
        if action == 'tune' and not reasons:
            check(lambda: tuning_conditions(cfg, conditions, task))
            from cgap_sbl_jadce.variants import SPATIAL_METHODS
            if task in SPATIAL_METHODS:
                from cgap_sbl_jadce.settings import validate_spatial_search
                check(lambda: validate_spatial_search(cfg, task))
    if task == 'glc_grid_study':
        from cgap_sbl_jadce.settings import grid_study_settings
        values = grid_study_settings(cfg)
        check(lambda: validate_communication_dataset(cfg, 'tuning', [(values['pilot_length'], values['snr_db'])]))
        check(lambda: validate_spatial_dependence_dataset(cfg, 'tuning'))
    if task == 'comparison':
        available = completed_methods(cfg, conditions)
        selected = methods if methods is not None else available
        if not selected:
            reasons.append('Complete at least one algorithm for these conditions, then select it here.')
        for method in selected:
            if method not in available:
                reasons.append(f'{ESTIMATORS[method].name} has no current complete scores for these conditions.')
    if task == 'convergence':
        check(lambda: validate_communication_dataset(cfg, 'evaluation', [condition] if condition else None))
        available = convergence_methods(cfg, condition)
        selected = methods if methods is not None else available
        if not selected:
            reasons.append('Complete an SBL-family run at this condition, then select it here.')
        for method in selected:
            if method not in available:
                reasons.append(f'Complete a compatible evaluation run for {ESTIMATORS[method].name}.')
    return reasons
