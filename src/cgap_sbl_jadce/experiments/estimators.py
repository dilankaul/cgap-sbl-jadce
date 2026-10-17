from functools import partial
from cgap_sbl_jadce.variants import SPATIAL_METHODS, parameters as variant_parameters
from copy import deepcopy
import hashlib
from dataclasses import dataclass
from typing import Callable
import json
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from cgap_sbl_jadce.file_cache import snapshot_cached
from cgap_sbl_jadce.settings import (solver_settings, tuning_settings, reference_settings, evaluation_settings, tuning_pairs, method_config, range_values, analysis_settings)
from cgap_sbl_jadce.algorithms.omp import mmv_omp
from cgap_sbl_jadce.algorithms.cosamp import mmv_cosamp
from cgap_sbl_jadce.algorithms.sbl import sbl
from cgap_sbl_jadce.communication_dataset import (
    validate_communication_dataset,
    load_communication_sample, load_tuning_communication,
)
from cgap_sbl_jadce.spatial_dependence_dataset import (
    spatial_dependence_definition, validate_spatial_dependence_dataset,
    load_spatial_dependence_matrix, load_tuning_spatial_dependence,
)
from cgap_sbl_jadce.experiments.tuning import threshold_curve
from cgap_sbl_jadce.metrics import (NMSE_SCOPE, nmse, oracle_sparsity_budget,
    precision_recall_f1, support_from_gamma, support_from_rows)
from cgap_sbl_jadce.io import save_csv, save_json, save_npz
from cgap_sbl_jadce.paths import result_path, task_dir, tuning_path, run_path
from cgap_sbl_jadce.provenance import save_task_snapshot
from cgap_sbl_jadce.reporting import print_task_report
from cgap_sbl_jadce.checkpoints import Checkpoint, checkpointed, is_resuming, archive_identity


def _sbl_options(cfg, method="sbl"):
    return solver_settings(cfg, method)


def _omp(r, C, parameters, cfg, **kwargs):
    return mmv_omp(r.Theta, r.Y, parameters['B_q'])


def _cosamp(r, C, parameters, cfg, **kwargs):
    solver = solver_settings(cfg, 'cosamp')
    return mmv_cosamp(r.Theta, r.Y, parameters['B_q'],
                      max_iter=solver['cosamp_max_iter'], tol=solver['cosamp_tol'],
                      lambda_reg=solver['cosamp_lambda_reg'])


def _sbl(r, C, parameters, cfg, **kwargs):
    return sbl(r.Theta, r.Y, r.sigma2, **_sbl_options(cfg), **kwargs)


def _spatial(r, C, parameters, cfg, method, **kwargs):
    from cgap_sbl_jadce.algorithms.cgap_variants import cgap_sbl_variant
    return cgap_sbl_variant(r.Theta, r.Y, r.sigma2, C, method,
                            **{name: parameters[name] for name in variant_parameters(method)},
                            **_sbl_options(cfg, method), **kwargs)


def _spatial_search(cfg, method):
    from cgap_sbl_jadce.settings import spatial_search_settings
    return spatial_search_settings(cfg, method)


def _glc_grid(cfg):
    from cgap_sbl_jadce.settings import grid_study_settings
    study = grid_study_settings(cfg)
    return [{'alpha': float(alpha), 'beta': float(beta)}
            for alpha in range_values(study, 'alpha') for beta in range_values(study, 'beta')]


@dataclass(frozen=True)
class Estimator:
    task: str
    name: str
    solve: Callable
    tuning_grid: Callable | None = None
    spatial: bool = False


ESTIMATORS = {
    'omp': Estimator('omp', 'MMV-OMP', _omp),
    'cosamp': Estimator('cosamp', 'MMV-CoSaMP', _cosamp),
    'sbl': Estimator('sbl', 'SBL', _sbl, lambda cfg: [{}]),
    **{method: Estimator(method, 'CGAP-SBL-' + method.upper(), partial(_spatial, method=method),
                          partial(_spatial_search, method=method), spatial=True) for method in SPATIAL_METHODS},
}


def tuning_definition(cfg, method):
    cfg = method_config(cfg, method)
    tune = tuning_settings(cfg, method)
    pair = (tune['reference_pilot_length'], tune['reference_snr_db'])
    pooled = tune['selection_mode'] == 'pooled'
    from cgap_sbl_jadce.communication_dataset import communication_condition_identity
    from cgap_sbl_jadce.compatibility import solver_revision
    from cgap_sbl_jadce.compatibility import _source_digest
    sources = ['experiments/tuning.py', 'experiments/search.py']
    if method in SPATIAL_METHODS:
        sources.extend(['experiments/spatial_tuning.py', 'variants.py'])
    definition = {'search_code': _source_digest(sources), 'method': ESTIMATORS[method].name, 'revision': 2, 'nmse_scope': NMSE_SCOPE,
        'condition': list(pair), 'tau_grid': {key: tune['tau_' + key] for key in ('start', 'stop', 'step')},
        'solver': solver_settings(cfg, method), 'code': solver_revision(method),
        'communication': None if pooled else communication_condition_identity(cfg, 'tuning', pair),
        'parameters': ESTIMATORS[method].tuning_grid(cfg), 'samples': cfg['run']['num_tuning_samples'],
        'selected_energy_format': 1}
    definition['row_norm_diagnostic'] = {'start': 0., 'stop': 1., 'step': .001,
                                             'score': 'norm(mu_i)/sqrt(M)'}
    if ESTIMATORS[method].spatial:
        import optuna
        definition['revision'] = 5
        definition['search_method'] = 'optuna_tpe'
        definition['optuna_version'] = optuna.__version__
        search = _spatial_search(cfg, method)
        definition['samples'] = search['samples']
        definition['sample_indices'] = list(range(search['samples']))
        definition['spatial_dependence'] = spatial_dependence_definition(cfg, 'tuning')
        definition['spatial_archive_sha256'] = archive_identity(task_dir(cfg['run']['output_dir'], 'spatial_dependence') / 'tuning', cfg['run']['num_tuning_samples'], ['matrix'])
    if pooled:
        from cgap_sbl_jadce.settings import pooled_assignment
        assignment = pooled_assignment(cfg)
        pairs = list(dict.fromkeys((row['L'], row['snr_db']) for row in assignment))
        definition.update(revision=6, objective='mean_top_S_gamma_f1', S=int(cfg['system']['S']),
                          assignment=assignment, samples=len(assignment),
                          communication=[communication_condition_identity(cfg, 'tuning', value) for value in pairs])
        definition['search_code'] = _source_digest(sources + ['experiments/estimators.py', 'settings.py', 'metrics.py'])
    return definition


def condition_config(cfg, condition, method=None):
    fixed_snr = evaluation_settings(cfg)['snr_db']
    cfg = method_config(cfg, method) if method else deepcopy(cfg)
    cfg['evaluation']['snr_db'] = fixed_snr
    cfg['communication'] = {**cfg.get('communication', {}), 'reference': dict(zip(
        ('reference_pilot_length', 'reference_snr_db'), condition))}
    if method and method in cfg['algorithms']:
        cfg['algorithms'][method]['tuning'].update(cfg['communication']['reference'])
    elif 'tuning' in cfg:
        cfg['tuning'].update(cfg['communication']['reference'])
    return cfg


def tuning_conditions(cfg, requested=None, method="glc"):
    cfg = method_config(cfg, method)
    pairs = tuning_pairs(cfg, method, include_optional=requested is not None)
    if tuning_settings(cfg, method)['selection_mode'] != 'per_condition':
        reference = (reference_settings(cfg)['reference_pilot_length'], reference_settings(cfg)['reference_snr_db'])
        if requested is not None and {tuple(pair) for pair in requested} != {reference}:
            raise ValueError('Reference tuning uses only the setup reference condition')
        return [reference]
    if requested is None:
        return pairs
    selected = list(dict.fromkeys(tuple(pair) for pair in requested))
    if not selected or not set(selected) <= set(pairs):
        raise ValueError('Choose saved tuning L/SNR conditions from this setup')
    return selected


@snapshot_cached
def load_selection(cfg, method, condition=None):
    if condition is not None and tuning_settings(cfg, method)['selection_mode'] == 'per_condition':
        cfg = condition_config(cfg, condition, method)
    cfg = method_config(cfg, method)
    path = tuning_path(cfg, method, 'selected.json')
    from cgap_sbl_jadce.tuning_results import pending_tuning_resets
    tune = tuning_settings(cfg, method)
    pair = (int(tune['reference_pilot_length']), float(tune['reference_snr_db']))
    if pair in pending_tuning_resets(cfg, method):
        raise ValueError('An interrupted tuning reset must finish through Resume on its original request.')
    spec = ESTIMATORS[method]
    if not path.exists():
        raise FileNotFoundError(f'Choose Algorithms > {spec.name} > Tune : {path}')
    selected = json.loads(path.read_text())
    if method in SPATIAL_METHODS and selected.get('search_method') != 'optuna_tpe':
        raise ValueError(f'{method.upper()} Run requires its own full Optuna TPE selection; paper grid and two-stage results cannot be used.')
    definition = tuning_definition(cfg, method)
    from cgap_sbl_jadce.compatibility import require_match
    require_match(selected.get('definition', {}), definition, f'{spec.name} tuning')
    return selected


def selection_digest(selected):
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def evaluation_selections(cfg, method, required):
    spec = ESTIMATORS[method]
    if not spec.tuning_grid:
        return {}
    selections = {}
    available = evaluation_conditions(cfg, method=method, include_optional=True)
    if tuning_settings(cfg, method)['selection_mode'] in {'reference', 'pooled'}:
        selected = load_selection(cfg, method)
        return {pair: selected for pair in dict.fromkeys(value[:2] for value in available)}
    for pair in dict.fromkeys(value[:2] for value in available):
        try:
            selections[pair] = load_selection(cfg, method, pair)
        except (OSError, ValueError):
            if pair in {(L, snr) for L, snr, _ in required}:
                raise
    return selections


def tune_estimator(cfg, method, show_progress=True):
    spec = ESTIMATORS[method]
    cfg = method_config(cfg, method)
    from cgap_sbl_jadce.settings import validate_spatial_search
    count = validate_spatial_search(cfg, method)['samples'] if method in SPATIAL_METHODS else cfg['run']['num_tuning_samples']
    pooled = tuning_settings(cfg, method)['selection_mode'] == 'pooled'
    if pooled:
        from cgap_sbl_jadce.settings import pooled_assignment
        assignment = pooled_assignment(cfg)
        pairs = list(dict.fromkeys((row['L'], row['snr_db']) for row in assignment))
        folder = validate_communication_dataset(cfg, 'tuning', pairs)
        communication = [load_communication_sample(folder, row['q'], (row['L'], row['snr_db']))[(row['L'], row['snr_db'])]
                         for row in assignment]
    else:
        communication = load_tuning_communication(cfg, count)
    matrices = (load_tuning_spatial_dependence(cfg, len(communication)) if spec.spatial
                else [None] * len(communication))
    checkpoint = Checkpoint(tuning_path(cfg, method, ''), tuning_definition(cfg, method))
    if method in SPATIAL_METHODS:
        from cgap_sbl_jadce.experiments.spatial_tuning import tune_spatial_optuna
        rows, curves = tune_spatial_optuna(cfg, communication, matrices, checkpoint, spec.solve, show_progress, method)
    else:
        from cgap_sbl_jadce.experiments.search import score_candidate
        rows, curves = [], []
        for index, parameters in enumerate(spec.tuning_grid(cfg)):
            row, curve = score_candidate(cfg, parameters, communication, matrices, checkpoint, spec.solve,
                                          range_values(tuning_settings(cfg, method), 'tau'), len(communication),
                                          show_progress, f'{spec.name} tuning', prefix=f'pair_{index:04d}', pooled=pooled)
            rows.append(row)
            curves.append(curve)
    from cgap_sbl_jadce.experiments.search import rank_score
    order = sorted(range(len(rows)), key=lambda i: rank_score(rows[i]), reverse=True)
    winner = order[0]
    best = rows[winner]
    if pooled:
        from cgap_sbl_jadce.experiments.search import candidate_key
        winner_parameters = {key: best[key] for key in variant_parameters(method)}
        winner_prefix = ('candidate_' + candidate_key(winner_parameters) if method in SPATIAL_METHODS else f'pair_{winner:04d}')
        gamma_scores = np.stack([checkpoint.load(f'{winner_prefix}_{q:04d}')[1]['gamma'] for q in range(count)])
        curves[winner] = threshold_curve(np.stack([r.a for r in communication]), gamma_scores,
                                        range_values(tuning_settings(cfg, method), 'tau'))
        calibrated = max(curves[winner], key=lambda point: (point['f1'], point['recall'], point['precision']))
    selected = {'definition': tuning_definition(cfg, method),
                'parameters': {**{key: best[key] for key in variant_parameters(method)}, 'tau': calibrated['tau'] if pooled else best['tau']},
                'tuning_scores': best,
                'reference_pilot_length': reference_settings(cfg)['reference_pilot_length'],
                'reference_snr_db': reference_settings(cfg)['reference_snr_db']}
    if method in SPATIAL_METHODS:
        selected.update(search_method='optuna_tpe')
    from cgap_sbl_jadce.experiments.search import candidate_key
    parameters = {key: best[key] for key in variant_parameters(method)}
    prefix = ('candidate_' + candidate_key(parameters) if method in SPATIAL_METHODS
              else f'pair_{winner:04d}')
    scores = np.stack([checkpoint.load(f'{prefix}_{q:04d}')[1]['row_norm']
                       for q in range(count)])
    norm_curve = threshold_curve(np.stack([r.a for r in communication[:count]]), scores,
                                 np.arange(1001, dtype=float) / 1000)
    norm_best = max(norm_curve, key=lambda point: (point['f1'], point['recall'], point['precision']))
    save_csv(tuning_path(cfg, method, 'row_norm_thresholds.csv'), norm_curve)
    save_json(tuning_path(cfg, method, 'row_norm_threshold_summary.json'), {
        'parameters': parameters, 'score': 'norm(mu_i)/sqrt(M)',
        'samples': count, 'best': norm_best, 'diagnostic_only': not pooled, 'used_for_parameter_search': False})
    selected['calibration'] = {'gamma_threshold': selected['parameters']['tau'], 'z_threshold': norm_best['tau'],
                               'source': 'pooled_tuning' if pooled else 'tuning', 'used_for_parameter_search': not pooled}
    if pooled:
        save_csv(tuning_path(cfg, method, 'assignment.csv'), assignment)
        save_npz(tuning_path(cfg, method, 'final_estimates.npz'), q=np.arange(count),
                 conditions=np.asarray([[row['L'], row['snr_db']] for row in assignment]),
                 a=np.stack([r.a for r in communication]), gamma=gamma_scores,
                 Z_hat=np.stack([checkpoint.load(f'{prefix}_{q:04d}')[1]['Z_hat'] for q in range(count)]))
        from cgap_sbl_jadce.metrics import support_from_scores
        cell_rows = []
        for L, snr in pairs:
            ids = [row['q'] for row in assignment if (row['L'], row['snr_db']) == (L, snr)]
            cell_scores = [precision_recall_f1(communication[q].a, support_from_scores(gamma_scores[q], int(cfg['system']['S']))) for q in ids]
            cell_rows.append(dict(L=L, snr_db=snr, samples=len(ids),
                                  **dict(zip(('precision', 'recall', 'f1'), np.mean(cell_scores, axis=0)))))
        save_csv(tuning_path(cfg, method, 'condition_scores.csv'), cell_rows)
    indices = list(dict.fromkeys(int(q) for q in cfg.get('figures', {}).get('tuning_indices', [])
                                 if 0 <= int(q) < count))
    save_npz(tuning_path(cfg, method, 'selected_channel_energy.npz'),
             indices=np.asarray(indices, dtype=int),
             conditions=np.asarray([[communication[q].Theta.shape[0], assignment[q]['snr_db'] if pooled else reference_settings(cfg)['reference_snr_db']] for q in indices]),
             original=np.asarray([np.sum(np.abs(communication[q].Z) ** 2, axis=1) for q in indices]),
             estimated=np.asarray([checkpoint.load(f'{prefix}_{q:04d}')[1]['row_energy'] for q in indices]))
    out = cfg['run']['output_dir']
    save_json(tuning_path(cfg, method, 'selected.json'), selected)
    save_csv(tuning_path(cfg, method, 'thresholds.csv'), curves[winner])
    folder = tuning_path(cfg, method, '')
    from cgap_sbl_jadce.figures.estimator_threshold_scores import plot_threshold_tuning
    from cgap_sbl_jadce.figures.spatial_optuna_tuning import plot_spatial_optuna_tuning
    from cgap_sbl_jadce.checkpoints import output_hashes
    plots = plot_threshold_tuning(cfg, method)
    if method in SPATIAL_METHODS:
        plots.extend(plot_spatial_optuna_tuning(cfg, method))
    names = ['selected.json', 'thresholds.csv', 'selected_channel_energy.npz',
             'row_norm_thresholds.csv', 'row_norm_threshold_summary.json']
    if pooled:
        names.extend(['assignment.csv', 'final_estimates.npz', 'condition_scores.csv'])
    if spec.spatial:
        names.extend(['trials.csv', 'study.sqlite3', 'sampler.pkl', 'search_progress.json', 'search_space.json'])
    files = [folder / name for name in names] + plots
    import os
    save_json(folder / 'summary.json', {
        'status': 'complete', 'selected': selected, 'candidate_count': len(rows), 'tau_candidates': len(curves[winner]),
        'outputs': {os.path.relpath(path, folder): digest for path, digest in zip(files, output_hashes(files).values())},
    })
    return selected


@checkpointed
def tune_estimator_stage(cfg, method, show_progress=True, conditions=None):
    spec = ESTIMATORS[method]
    if spec.tuning_grid is None:
        raise ValueError(f"{spec.name} does not require tuning.")
    cfg = method_config(cfg, method)
    pairs = tuning_conditions(cfg, conditions, method)
    from cgap_sbl_jadce.settings import validate_spatial_search
    if method in SPATIAL_METHODS:
        validate_spatial_search(cfg, method)
    range_values(tuning_settings(cfg, method), 'tau')
    required = pairs
    if tuning_settings(cfg, method)['selection_mode'] == 'pooled':
        from cgap_sbl_jadce.settings import pooled_assignment
        required = list(dict.fromkeys((row['L'], row['snr_db']) for row in pooled_assignment(cfg)))
    validate_communication_dataset(cfg, 'tuning', required)
    if spec.spatial:
        validate_spatial_dependence_dataset(cfg, 'tuning')
    definitions = [tuning_definition(condition_config(cfg, pair, method), method) for pair in pairs]
    from cgap_sbl_jadce.tuning_results import reset_tuning_results
    from cgap_sbl_jadce.figures.estimator_parameter_sweeps import plot_tuning_parameter_sweeps
    reset_tuning_results(cfg, method, pairs, definitions, is_resuming())
    save_task_snapshot(cfg, spec.task, status='started', action='tune', method=spec.name)
    selections = []
    refreshed = False
    for pair in pairs:
        condition_cfg = condition_config(cfg, pair, method)
        summary_path = tuning_path(condition_cfg, method, 'summary.json')
        if is_resuming() and summary_path.exists():
            saved = json.loads(summary_path.read_text())
            try:
                current = load_selection(condition_cfg, method)
                from cgap_sbl_jadce.checkpoints import output_hashes
                root = summary_path.parent
                files = [root / name for name in saved.get('outputs', {})]
                hashes = {str(path.relative_to(root)): digest for path, digest in zip(files, output_hashes(files).values())}
                if saved.get('status') == 'complete' and hashes == saved.get('outputs'):
                    from cgap_sbl_jadce.checkpoints import prune_owned_checkpoints
                    prune_owned_checkpoints(root / '.checkpoints', tuning_definition(condition_cfg, method))
                    selections.append(current)
                    continue
            except (OSError, ValueError, KeyError):
                pass
        selections.append(tune_estimator(condition_cfg, method, show_progress))
        plot_tuning_parameter_sweeps(cfg, method)
        if method in SPATIAL_METHODS:
            from cgap_sbl_jadce.checkpoints import prune_owned_checkpoints
            prune_owned_checkpoints(tuning_path(condition_cfg, method, '') / '.checkpoints',
                                    tuning_definition(condition_cfg, method))
        refreshed = True
    if not refreshed:
        plot_tuning_parameter_sweeps(cfg, method)
    selected = selections[0] if len(selections) == 1 else {'conditions': selections}
    save_task_snapshot(cfg, spec.task, status='complete', action='tune',
                       method=spec.name, selected_parameters=selected)
    print_task_report(f'{spec.name} TUNING',
                      {'tuned conditions': len(selections), 'selections': [item['parameters'] for item in selections]},
                      [tuning_path(condition_config(cfg, pair, method), method, 'selected.json') for pair in pairs])
    return selected


def evaluation_conditions(cfg, selected=None, method=None, include_optional=False):
    settings = evaluation_settings(cfg)
    L0 = int(settings['pilot_length'])
    snr0 = float(settings['snr_db'])
    conditions = ([(L0, float(snr), 'snr_sweep') for snr in settings['snr_values']]
                  + [(int(L), snr0, 'pilot_sweep') for L in settings['pilot_lengths']])
    if include_optional or selected is not None:
        from cgap_sbl_jadce.additional_conditions import additional_conditions
        existing = {value[:2] for value in conditions}
        conditions.extend((L, snr, 'condition') for L, snr in additional_conditions(cfg, method) if (L, snr) not in existing)
    if selected is None:
        return conditions
    pairs = {tuple(pair) for pair in selected}
    if not pairs or not pairs <= {(L, snr) for L, snr, _ in conditions}:
        raise ValueError('Choose at least one L/SNR pair from the setup evaluation conditions.')
    return [condition for condition in conditions if condition[:2] in pairs]


def evaluation_definition(cfg, method, condition=None):
    from cgap_sbl_jadce.compatibility import solver_revision
    from cgap_sbl_jadce.communication_dataset import communication_condition_identity
    pairs = [tuple(condition)] if condition is not None else list(dict.fromkeys(value[:2] for value in evaluation_conditions(cfg)))
    definition = {'method': ESTIMATORS[method].name, 'revision': 4, 'nmse_scope': NMSE_SCOPE,
        'solver': solver_settings(cfg, method), 'code': solver_revision(method),
        'samples': cfg['run']['num_evaluation_samples'],
        'communication': [communication_condition_identity(cfg, 'evaluation', pair) for pair in pairs]}
    if ESTIMATORS[method].tuning_grid:
        definition['selections'] = [load_selection(cfg, method, pair) for pair in pairs]
    if ESTIMATORS[method].spatial:
        definition['spatial_dependence'] = spatial_dependence_definition(cfg, 'evaluation')
        definition['spatial_archive_sha256'] = archive_identity(task_dir(cfg['run']['output_dir'], 'spatial_dependence') / 'evaluation', cfg['run']['num_evaluation_samples'], ['matrix'])
    return definition


def evaluation_matches(saved, cfg, method, pair):
    return saved == evaluation_definition(cfg, method, condition=pair)


def _aggregate(df):
    keys = ['algorithm', 'L', 'snr_db', 'sweep']
    metrics = ['precision', 'recall', 'f1', 'nmse', 'runtime_s'] + [key for key in ('nmse_unmasked', 'nmse_support_masked') if key in df]
    aggregate = df.groupby(keys, as_index=False)[metrics].mean()
    if 'nmse_mode' in df:
        aggregate = aggregate.merge(df.groupby(keys, as_index=False).nmse_mode.first(), on=keys)
    aggregate['nmse_scope'] = NMSE_SCOPE
    runtime = df.groupby(keys)['runtime_s'].agg(
        samples='count', mean_s='mean', std_s='std', median_s='median',
        min_s='min', max_s='max', total_s='sum', p95_s=lambda x: x.quantile(.95)).reset_index()
    return aggregate, runtime


@checkpointed
def run_estimator_stage(cfg, method, show_progress=True, conditions=None, resume=False):
    from cgap_sbl_jadce.condition_results import (completed_conditions, rebuild_algorithm_results, reset_run_results)
    from cgap_sbl_jadce.figures.estimator_condition import plot_condition_scores
    from cgap_sbl_jadce.checkpoints import output_hashes
    resume = is_resuming()
    spec = ESTIMATORS[method]
    out = cfg['run']['output_dir']
    communication_folder = validate_communication_dataset(cfg, 'evaluation', [value[:2] for value in evaluation_conditions(cfg, conditions, method=method)])
    C_folder = validate_spatial_dependence_dataset(cfg, 'evaluation') if spec.spatial else None
    requested = evaluation_conditions(cfg, conditions, method=method)
    selections = evaluation_selections(cfg, method, requested)
    selected = next(iter(selections.values()), None)
    save_task_snapshot(cfg, spec.task, status='started', action='run', method=spec.name)
    count = int(cfg['run']['num_evaluation_samples'])
    detector_settings = analysis_settings(cfg, method)
    gamma_count = min(int(cfg['evaluation'].get('gamma_distribution_samples', 0)), count)
    pairs = list(dict.fromkeys(value[:2] for value in requested))
    definitions = [evaluation_definition(cfg, method, condition=pair) for pair in pairs]
    reset_run_results(cfg, method, pairs, definitions, resume)
    done = {pair for pair, _, _ in completed_conditions(cfg, method, selections)}
    for pair, definition in zip(tqdm(pairs, desc=f'{spec.name} conditions', unit='condition', disable=not show_progress), definitions):
        if resume and pair in done:
            continue
        L, snr = pair
        folder = run_path(cfg, method, condition=pair)
        parameters = selections[pair]['parameters'] if spec.tuning_grid else {}
        signature = selection_digest(selections[pair]) if pair in selections else None
        checkpoint = Checkpoint(folder, {'definition': definition, 'condition': list(pair),
                                         'selection_digest': signature, 'gamma_count': gamma_count,
                                         'energy_indices': cfg.get('figures', {}).get('evaluation_indices', [])})
        save_json(folder / 'summary.json', {'status': 'running', 'condition': list(pair), 'definition': definition, 'parameters': parameters})
        rows, gammas, masks = [], [], []
        final_estimates, final_gammas, final_masks = [], [], []
        energy_indices, original_energy, estimated_energy = [], [], []
        views = [value[2] for value in requested if value[:2] == pair]
        for q in tqdm(range(count), desc=f'{spec.name} L={L:g} SNR={snr:g} dB',
                      unit='realization', disable=not show_progress):
            cached = checkpoint.load(f'{q:04d}') if resume else None
            if cached is not None:
                metadata, arrays = cached
                row = metadata['row']
            else:
                r = load_communication_sample(communication_folder, q, pair)[pair]
                C = load_spatial_dependence_matrix(C_folder, q) if C_folder else None
                S_q = int(r.a.sum())
                solver_parameters = parameters if spec.tuning_grid else {'B_q': oracle_sparsity_budget(S_q, L, len(r.a))}
                start = time.perf_counter()
                result = spec.solve(r, C, solver_parameters, cfg)
                runtime = time.perf_counter() - start
                estimate = result.mu if spec.tuning_grid else result
                from cgap_sbl_jadce.experiments.postprocessing import detection_support
                support = detection_support(estimate, result.gamma if spec.tuning_grid else None,
                    detector_settings, selections.get(pair), solver_parameters.get('B_q'))
                precision, recall, f1 = precision_recall_f1(r.a, support)
                row = {'q': q, 'algorithm': spec.name, 'S_q': S_q, 'L': L, 'snr_db': snr,
                       'precision': precision, 'recall': recall, 'f1': f1,
                       'nmse': nmse(r.Z, estimate, r.a), 'nmse_scope': NMSE_SCOPE,
                       'nmse_unmasked': nmse(r.Z, estimate, r.a),
                       'nmse_support_masked': nmse(r.Z, estimate * support[:, None], r.a),
                       'iterations': result.iterations if spec.tuning_grid else (solver_parameters['B_q'] if method == 'omp' else np.nan),
                       'runtime_s': runtime}
                row['nmse_mode'] = detector_settings['nmse_mode']
                if row['nmse_mode'] == 'support_masked':
                    row['nmse'] = row['nmse_support_masked']
                if not spec.tuning_grid:
                    row.update(B_q=solver_parameters['B_q'], oracle_capped=solver_parameters['B_q'] < S_q)
                arrays = {'a': r.a, 'Z_hat': estimate}
                if spec.tuning_grid:
                    arrays['gamma'] = result.gamma
                if q in cfg.get('figures', {}).get('evaluation_indices', []):
                    arrays.update(original_energy=np.sum(np.abs(r.Z) ** 2, axis=1),
                                  estimated_energy=np.sum(np.abs(estimate) ** 2, axis=1))
                checkpoint.save(f'{q:04d}', {'row': row}, **arrays)
            if cached is not None:
                from cgap_sbl_jadce.experiments.postprocessing import detection_support
                r = load_communication_sample(communication_folder, q, pair)[pair]
                support = detection_support(arrays['Z_hat'], arrays.get('gamma'), detector_settings,
                                            selections.get(pair), int(row['B_q']) if 'B_q' in row else None)
                precision, recall, f1 = precision_recall_f1(r.a, support)
                unmasked = nmse(r.Z, arrays['Z_hat'], r.a)
                masked = nmse(r.Z, arrays['Z_hat'] * support[:, None], r.a)
                row.update(precision=precision, recall=recall, f1=f1, nmse_unmasked=unmasked,
                           nmse_support_masked=masked, nmse_mode=detector_settings['nmse_mode'],
                           nmse=masked if detector_settings['nmse_mode'] == 'support_masked' else unmasked)
            final_estimates.append(arrays['Z_hat'])
            final_masks.append(arrays['a'])
            if spec.tuning_grid:
                final_gammas.append(arrays['gamma'])
            rows.extend({**row, 'sweep': view} for view in views)
            if 'original_energy' in arrays:
                energy_indices.append(q)
                original_energy.append(arrays['original_energy'])
                estimated_energy.append(arrays['estimated_energy'])
            if 'gamma' in arrays and q < gamma_count:
                masks.append(arrays['a']); gammas.append(arrays['gamma'])
        final_arrays = {'q': np.arange(count), 'a': np.stack(final_masks), 'Z_hat': np.stack(final_estimates)}
        if final_gammas:
            final_arrays['gamma'] = np.stack(final_gammas)
        save_npz(folder / 'final_estimates.npz', **final_arrays)
        save_csv(folder / 'per_sample.csv', rows)
        aggregate, runtime = _aggregate(pd.DataFrame(rows))
        save_csv(folder / 'aggregate.csv', aggregate)
        save_csv(folder / 'runtime_summary.csv', runtime)
        files = [folder / name for name in ('per_sample.csv', 'aggregate.csv', 'runtime_summary.csv', 'final_estimates.npz')]
        if gammas:
            save_npz(folder / 'gamma_distribution.npz', a=np.stack(masks), gamma=np.stack(gammas))
            files.append(folder / 'gamma_distribution.npz')
        save_npz(folder / 'selected_channel_energy.npz', indices=np.asarray(energy_indices, dtype=int),
                 original=np.asarray(original_energy), estimated=np.asarray(estimated_energy))
        files.append(folder / 'selected_channel_energy.npz')
        files.extend(plot_condition_scores(cfg, method, pair))
        save_json(folder / 'summary.json', {
            'status': 'complete', 'condition': list(pair), 'definition': definition,
            'parameters': parameters, 'selection_digest': signature, 'Q': count,
            'gamma_samples': len(gammas), 'nmse_scope': NMSE_SCOPE,
            'final_estimate_samples': count,
            'analysis': analysis_settings(cfg, method),
            'outputs': {str(path.relative_to(folder)): digest
                        for path, digest in zip(files, output_hashes(files).values())},
        })

        checkpoint.clear(prune=True)
        rebuild_algorithm_results(cfg, method, selections=selections)
    summary = rebuild_algorithm_results(cfg, method, selections=selections)
    summary['requested_conditions'] = [list(value) for value in requested]
    save_json(result_path(out, method, 'run_summary.json'), summary)
    save_task_snapshot(cfg, spec.task, status=summary['status'], action='run', method=spec.name, selected_parameters=selected)
    print_task_report(spec.name, {'evaluation realizations': count,
                                 'completed conditions': len(summary['completed_pairs']),
                                 'total conditions': summary['total_conditions'], 'NMSE scope': NMSE_SCOPE},
                      [run_path(cfg, method), result_path(out, method, 'run_summary.json')])
    return summary


def completed_methods(cfg, conditions=None):
    from cgap_sbl_jadce.condition_results import completed_conditions
    requested = {value[:2] for value in evaluation_conditions(cfg, conditions)}
    available = []
    for method in ESTIMATORS:
        try:
            current = {pair for pair, _, _ in completed_conditions(cfg, method)}
            if requested <= current:
                available.append(method)
        except (OSError, ValueError, KeyError):
            continue
    return available


def convergence_methods(cfg, condition=None):
    condition = condition or (cfg['evaluation']['pilot_length'], evaluation_settings(cfg)['snr_db'])
    try:
        completed = completed_methods(cfg, [condition])
        validate_communication_dataset(cfg, 'evaluation', [condition])
    except (OSError, ValueError):
        return []
    available = []
    for method in completed:
        spec = ESTIMATORS[method]
        if not spec.tuning_grid:
            continue
        try:
            load_selection(cfg, method, condition)
            if spec.spatial:
                validate_spatial_dependence_dataset(cfg, 'evaluation')
        except (OSError, ValueError):
            continue
        available.append(method)
    return available


def analysis_methods(cfg, kind, methods=None, conditions=None, condition=None):
    available = completed_methods(cfg, conditions) if kind == 'comparison' else convergence_methods(cfg, condition)
    selected = list(dict.fromkeys(methods)) if methods is not None else available
    if not selected or not set(selected) <= set(ESTIMATORS):
        raise ValueError('Select at least one available algorithm.')
    return selected


def comparison_stage(cfg, show_progress=True, conditions=None, methods=None):
    out = cfg['run']['output_dir']
    requested = evaluation_conditions(cfg, conditions)
    methods = analysis_methods(cfg, 'comparison', methods, conditions)
    datasets = []
    for method in methods:
        spec = ESTIMATORS[method]
        summary_path = result_path(out, method, 'run_summary.json')
        if not summary_path.exists():
            raise FileNotFoundError(f'Run {spec.name} for these conditions first.')
        summary = json.loads(summary_path.read_text())
        if method not in completed_methods(cfg, conditions):
            raise ValueError(f'Incompatible {spec.name} results. Rerun {spec.name}.')
        from cgap_sbl_jadce.condition_results import completed_conditions
        df = pd.concat([frame for pair, _, frame in completed_conditions(cfg, method) if pair in {value[:2] for value in requested}], ignore_index=True)
        expected = {(q, L, snr, sweep) for q in range(cfg['run']['num_evaluation_samples'])
                    for L, snr, sweep in requested}
        df = df[df.apply(lambda row: (row.L, row.snr_db, row.sweep) in requested, axis=1)]
        actual = set(df[['q', 'L', 'snr_db', 'sweep']].itertuples(index=False, name=None))
        if (len(df) != len(expected) or actual != expected or not df['algorithm'].eq(spec.name).all()
                or 'nmse_scope' not in df or not df['nmse_scope'].eq(NMSE_SCOPE).all()):
            raise ValueError(f'Incomplete or incompatible {spec.name} scores. Rerun {spec.name}.')
        from cgap_sbl_jadce.experiments.postprocessing import rescore_condition
        selections = evaluation_selections(cfg, method, requested)
        df = pd.concat([rescore_condition(cfg, method, pair, df[(df.L == pair[0]) & (df.snr_db == pair[1])], selections.get(pair), show_progress=show_progress)
                        for pair in dict.fromkeys(value[:2] for value in requested)], ignore_index=True)
        datasets.append(df)
    df = pd.concat(datasets, ignore_index=True)
    folder = task_dir(out, 'comparison')
    folder.mkdir(parents=True, exist_ok=True)
    aggregate, runtime = _aggregate(df)
    save_csv(folder / 'per_sample.csv', df)
    save_csv(folder / 'aggregate.csv', aggregate)
    save_csv(folder / 'runtime_summary.csv', runtime)
    from cgap_sbl_jadce.experiments.postprocessing import analysis_revision
    summary = {'status': 'complete', 'nmse_scope': NMSE_SCOPE, 'analysis_code': analysis_revision(),
               'analysis': {method: analysis_settings(cfg, method) for method in methods},
               'conditions': [list(condition) for condition in requested], 'methods': methods,
               'algorithms': [ESTIMATORS[method].name for method in methods], 'metric_rows': len(df),
               'definitions': {method: [evaluation_definition(cfg, method, condition=pair) for pair in dict.fromkeys(value[:2] for value in requested)] for method in methods}}
    save_json(folder / 'summary.json', summary)
    save_task_snapshot(cfg, 'comparison', status='complete')
    print_task_report('ALGORITHM COMPARISON', {key: value for key, value in summary.items() if key != 'definitions'}, [folder])
    return summary


@checkpointed
def convergence_stage(cfg, show_progress=True, methods=None, realization_index=None, condition=None):
    out = cfg['run']['output_dir']
    q = int(cfg['evaluation'].get('convergence_sample_offset', 0) if realization_index is None else realization_index)
    if not 0 <= q < cfg['run']['num_evaluation_samples']:
        raise ValueError('Convergence realization index is outside the evaluation dataset.')
    L, snr = condition or (int(cfg['evaluation']['pilot_length']), float(evaluation_settings(cfg)['snr_db']))
    r = load_communication_sample(validate_communication_dataset(cfg, 'evaluation', [(L, snr)]), q, (L, snr))[(L, snr)]
    methods = analysis_methods(cfg, 'convergence', methods, condition=(L, snr))
    if not set(methods) <= set(convergence_methods(cfg, (L, snr))):
        raise ValueError('Convergence requires completed SBL-family evaluation at this condition.')
    C = load_spatial_dependence_matrix(validate_spatial_dependence_dataset(cfg, 'evaluation'), q) if any(ESTIMATORS[method].spatial for method in methods) else None
    histories, scores = {}, {}
    selections = {method: load_selection(cfg, method, (L, snr)) for method in methods if ESTIMATORS[method].tuning_grid}
    checkpoint = Checkpoint(task_dir(out, 'convergence'), {'q': q, 'L': L, 'snr': snr, 'selections': selections,
        'methods': {method: evaluation_definition(cfg, method, condition=(L, snr)) for method in methods}})
    for method in tqdm(methods, desc='Convergence', unit='algorithm', disable=not show_progress):
        spec = ESTIMATORS[method]
        cached = checkpoint.load(method)
        if cached is not None:
            metadata, arrays = cached
        else:
            result = spec.solve(r, C if spec.spatial else None, selections[method]['parameters'], cfg, keep_history=True, Z_true=r.Z)
            arrays = {f'{method}_gamma_history': result.gamma_history, f'{method}_nmse_history': result.nmse_history,
                      f'{method}_iteration_history': np.arange(1, len(result.nmse_history) + 1)}
            if spec.spatial:
                arrays[f'{method}_lambda_history'] = result.lambda_history
            metadata = {'final_nmse': nmse(r.Z, result.mu, r.a), 'iterations': result.iterations, 'converged': result.converged}
            checkpoint.save(method, metadata, **arrays)
        histories.update(arrays)
        scores[method] = metadata
    save_npz(result_path(out, 'convergence', 'convergence.npz'), **histories)
    summary = {'status': 'complete', 'q': q, 'L': L, 'snr_db': snr, 'methods': methods, 'nmse_scope': NMSE_SCOPE, 'scores': scores,
               'definitions': {method: evaluation_definition(cfg, method, condition=(L, snr)) for method in methods}}
    save_json(result_path(out, 'convergence', 'summary.json'), summary)
    save_task_snapshot(cfg, 'convergence', status='complete', sample_index=q)
    print_task_report('CONVERGENCE', {key: value for key, value in summary.items() if key != 'definitions'}, [task_dir(out, 'convergence')])
    return summary
