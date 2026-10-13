"""Actual data dependencies for independently selected workflow actions."""
import json
from pathlib import Path

from cgap_sbl_jadce.config import resolve_activity_kappa
from cgap_sbl_jadce.communication_dataset import validate_communication_dataset
from cgap_sbl_jadce.spatial_dependence_dataset import validate_spatial_dependence_dataset
from cgap_sbl_jadce.paths import result_path
from cgap_sbl_jadce.experiments.estimators import (ESTIMATORS, evaluation_conditions, evaluation_selections,
    tuning_conditions, load_selection, completed_methods, convergence_methods)


def blocking_reasons(cfg, task, action='run', conditions=None, methods=None, condition=None):
    reasons = []
    def check(operation):
        try:
            operation()
        except (OSError, ValueError, KeyError) as error:
            reasons.append(str(error))

    def activity():
        for split in ('tuning', 'evaluation'):
            path = result_path(cfg['run']['output_dir'], 'activity', f'{split}_activity.npz')
            if not path.exists():
                raise FileNotFoundError(f'Generate {split} activity first.')

    if task == 'activity':
        check(lambda: resolve_activity_kappa(cfg))
    if task in {'communication', 'spatial_dependence'}:
        check(activity)
    if task in ESTIMATORS:
        spec = ESTIMATORS[task]
        split = 'tuning' if action == 'tune' else 'evaluation'
        check(lambda: validate_communication_dataset(cfg, split))
        if spec.spatial:
            check(lambda: validate_spatial_dependence_dataset(cfg, split))
        if action == 'run' and spec.tuning_grid and not reasons:
            check(lambda: evaluation_selections(cfg, task, evaluation_conditions(cfg, conditions)))
        if action == 'tune' and not reasons:
            check(lambda: tuning_conditions(cfg, conditions))
    if task == 'comparison':
        available = completed_methods(cfg, conditions)
        selected = methods if methods is not None else available
        if not selected:
            reasons.append('Complete at least one algorithm for these conditions, then select it here.')
        for method in selected:
            if method not in available:
                reasons.append(f'{ESTIMATORS[method].name} has no current complete scores for these conditions.')
    if task == 'convergence':
        check(lambda: validate_communication_dataset(cfg, 'evaluation'))
        available = convergence_methods(cfg, condition)
        selected = methods if methods is not None else available
        if not selected:
            reasons.append('Complete SBL or CGAP-SBL-ANC at this condition, then select it here.')
        for method in selected:
            if method not in available:
                reasons.append(f'Complete a compatible evaluation run for {ESTIMATORS[method].name}.')
    return reasons
