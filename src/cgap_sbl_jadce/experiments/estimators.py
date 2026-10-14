"""Independent estimator tasks over the same saved communication realizations."""
from copy import deepcopy
import hashlib
from dataclasses import dataclass
from typing import Callable
import json
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from cgap_sbl_jadce.settings import (solver_settings, tuning_settings, reference_settings, evaluation_settings, tuning_pairs, method_config, range_values)
from cgap_sbl_jadce.algorithms.omp import mmv_omp
from cgap_sbl_jadce.algorithms.cosamp import mmv_cosamp
from cgap_sbl_jadce.algorithms.sbl import sbl
from cgap_sbl_jadce.algorithms.cgap_sbl import cgap_sbl_anc
from cgap_sbl_jadce.communication_dataset import (
    communication_definition, validate_communication_dataset,
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
    return mmv_cosamp(r.Theta, r.Y, parameters['B_q'],
                      max_iter=solver_settings(cfg, 'cosamp')['cosamp_max_iter'], tol=solver_settings(cfg, 'cosamp')['cosamp_tol'])


def _sbl(r, C, parameters, cfg, **kwargs):
    return sbl(r.Theta, r.Y, r.sigma2, **_sbl_options(cfg), **kwargs)


def _anc(r, C, parameters, cfg, **kwargs):
    return cgap_sbl_anc(r.Theta, r.Y, r.sigma2, C,
                        parameters['alpha'], parameters['beta'], **_sbl_options(cfg, 'anc'), **kwargs)


def _anc_grid(cfg):
    return [{'alpha': float(alpha), 'beta': float(beta)}
            for alpha in range_values(tuning_settings(cfg, 'anc'), 'alpha') for beta in range_values(tuning_settings(cfg, 'anc'), 'beta')]


@dataclass(frozen=True)
class Estimator:
    task: str
    name: str
    solve: Callable
    tuning_grid: Callable | None = None
    spatial: bool = False


# A later GL variant supplies a solver and tuning grid here and reuses the workflow.
ESTIMATORS = {
    'omp': Estimator('omp', 'MMV-OMP', _omp),
    'cosamp': Estimator('cosamp', 'MMV-CoSaMP', _cosamp),
    'sbl': Estimator('sbl', 'SBL', _sbl, lambda cfg: [{}]),
    'anc': Estimator('anc', 'CGAP-SBL-ANC', _anc, _anc_grid, spatial=True),
}


def legacy_tuning_definition(cfg, method):
    spec = ESTIMATORS[method]
    definition = {'method': spec.name, 'revision': 1, 'nmse_scope': NMSE_SCOPE,
                  'condition': [reference_settings(cfg)['reference_pilot_length'], reference_settings(cfg)['reference_snr_db']],
                  'tau_grid': {key: tuning_settings(cfg, method)['tau_' + key] for key in ('start', 'stop', 'step')},
                  'algorithms': legacy_solver_config(cfg, method),
                  'communication': communication_definition(cfg, 'tuning'),
                  'parameters': spec.tuning_grid(cfg), 'samples': cfg['run']['num_tuning_samples'],
                  'archive_sha256': archive_identity(task_dir(cfg['run']['output_dir'], 'communication') / 'tuning',
                      cfg['run']['num_tuning_samples'], ['channels.npz', 'pilots.npz', 'observations.npz'])}
    if spec.spatial:
        definition['spatial_dependence'] = spatial_dependence_definition(cfg, 'tuning')
        definition['spatial_archive_sha256'] = archive_identity(task_dir(cfg['run']['output_dir'], 'spatial_dependence') / 'tuning', cfg['run']['num_tuning_samples'], ['matrix'])
    return definition


def legacy_solver_config(cfg, method):
    return {**solver_settings(cfg, method if method in {'sbl', 'anc'} else 'sbl'), **solver_settings(cfg, 'cosamp')}


def tuning_definition(cfg, method):
    cfg = method_config(cfg, method)
    tune = tuning_settings(cfg, method)
    pair = (tune['reference_pilot_length'], tune['reference_snr_db'])
    from cgap_sbl_jadce.communication_dataset import communication_condition_identity
    from cgap_sbl_jadce.compatibility import solver_revision
    definition = {'method': ESTIMATORS[method].name, 'revision': 2, 'nmse_scope': NMSE_SCOPE,
        'condition': list(pair), 'tau_grid': {key: tune['tau_' + key] for key in ('start', 'stop', 'step')},
        'solver': solver_settings(cfg, method), 'code': solver_revision(method),
        'communication': communication_condition_identity(cfg, 'tuning', pair),
        'parameters': ESTIMATORS[method].tuning_grid(cfg), 'samples': cfg['run']['num_tuning_samples']}
    if ESTIMATORS[method].spatial:
        definition['spatial_dependence'] = spatial_dependence_definition(cfg, 'tuning')
        definition['spatial_archive_sha256'] = archive_identity(task_dir(cfg['run']['output_dir'], 'spatial_dependence') / 'tuning', cfg['run']['num_tuning_samples'], ['matrix'])
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


def tuning_conditions(cfg, requested=None, method="anc"):
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


def load_selection(cfg, method, condition=None, expected=None):
    if condition is not None and tuning_settings(cfg, method)['selection_mode'] == 'per_condition':
        cfg = condition_config(cfg, condition, method)
    cfg = method_config(cfg, method)
    path = tuning_path(cfg, method, 'selected.json')
    spec = ESTIMATORS[method]
    if not path.exists():
        raise FileNotFoundError(f'Choose Algorithms > {spec.name} > Tune : {path}')
    selected = json.loads(path.read_text())
    definition = tuning_definition(cfg, method)
    if selected.get('definition', {}).get('revision') == 1:
        from cgap_sbl_jadce.compatibility import legacy_definition_matches
        matches = legacy_definition_matches(selected['definition'], legacy_tuning_definition(cfg, method), cfg, method)
    else:
        matches = selected.get('definition') == definition
    if not matches:
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
    if tuning_settings(cfg, method)['selection_mode'] == 'reference':
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
    communication = load_tuning_communication(cfg, cfg['run']['num_tuning_samples'])
    matrices = (load_tuning_spatial_dependence(cfg, len(communication)) if spec.spatial
                else [None] * len(communication))
    checkpoint = Checkpoint(tuning_path(cfg, method, ''), tuning_definition(cfg, method))
    rows, curves = [], []
    for pair_index, parameters in enumerate(tqdm(spec.tuning_grid(cfg), desc=f'{spec.name} tuning',
                           unit='pair' if spec.spatial else 'setup', disable=not show_progress)):
        gammas, errors, iterations = [], [], []
        for q, (r, C) in enumerate(zip(communication, matrices)):
            key = f'pair_{pair_index:04d}_{q:04d}'
            cached = checkpoint.load(key)
            if cached is None:
                result = spec.solve(r, C, parameters, cfg)
                metadata = {'nmse': nmse(r.Z, result.mu, r.a), 'iterations': result.iterations}
                arrays = {'gamma': result.gamma}
                checkpoint.save(key, metadata, **arrays)
            else:
                metadata, arrays = cached
            gammas.append(arrays['gamma'])
            errors.append(metadata['nmse'])
            iterations.append(metadata['iterations'])
        tune = tuning_settings(cfg, method)
        candidates = (None if (tune['tau_start'], tune['tau_stop'], tune['tau_step']) == (0., .2, .001)
                      else range_values(tune, 'tau'))
        curve = threshold_curve(np.stack([r.a for r in communication]), np.asarray(gammas), tau_values=candidates)
        best = max(curve, key=lambda point: (point['f1'], point['recall'], point['precision']))
        rows.append({**parameters, **best, 'nmse': float(np.nanmean(errors)),
                     'iterations': float(np.mean(iterations))})
        curves.append(curve)
    order = sorted(range(len(rows)), key=lambda i: (rows[i]['f1'], -rows[i]['nmse'], rows[i]['recall']), reverse=True)
    winner = order[0]
    best = rows[winner]
    selected = {'definition': tuning_definition(cfg, method),
                'parameters': {**spec.tuning_grid(cfg)[winner], 'tau': best['tau']},
                'tuning_scores': best,
                'reference_pilot_length': reference_settings(cfg)['reference_pilot_length'],
                'reference_snr_db': reference_settings(cfg)['reference_snr_db']}
    out = cfg['run']['output_dir']
    save_json(tuning_path(cfg, method, 'selected.json'), selected)
    save_csv(tuning_path(cfg, method, 'thresholds.csv'), curves[winner])
    if spec.spatial:
        save_csv(tuning_path(cfg, method, 'alpha_beta.csv'), [rows[i] for i in order])
        save_csv(tuning_path(cfg, method, 'alpha_beta_tau_scores.csv'),
                 [{**parameters, **point} for parameters, curve in zip(spec.tuning_grid(cfg), curves)
                  for point in curve])
    folder = tuning_path(cfg, method, '')
    from cgap_sbl_jadce.figures.estimator_threshold_scores import plot_threshold_tuning
    from cgap_sbl_jadce.figures.anc_parameter_tuning import plot_alpha_beta_tau_tuning
    from cgap_sbl_jadce.checkpoints import output_hashes
    plots = plot_threshold_tuning(cfg, method)
    if method == 'anc':
        plots.extend(plot_alpha_beta_tau_tuning(cfg))
    names = ['selected.json', 'thresholds.csv']
    if spec.spatial:
        names.extend(['alpha_beta.csv', 'alpha_beta_tau_scores.csv'])
    files = [folder / name for name in names] + plots
    import os
    save_json(folder / 'summary.json', {
        'status': 'complete', 'selected': selected, 'grid_pairs': len(rows), 'tau_candidates': len(curves[winner]),
        'outputs': {os.path.relpath(path, folder): digest for path, digest in zip(files, output_hashes(files).values())},
    })
    checkpoint.clear(prune=True)
    return selected



@checkpointed
def tune_estimator_stage(cfg, method, show_progress=True, conditions=None):
    """Tune one algorithm using only tuning realizations; save its selection."""
    spec = ESTIMATORS[method]
    if spec.tuning_grid is None:
        raise ValueError(f"{spec.name} does not require tuning.")
    save_task_snapshot(cfg, spec.task, status='started', action='tune', method=spec.name)
    cfg = method_config(cfg, method)
    pairs = tuning_conditions(cfg, conditions, method)
    selections = []
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
                    selections.append(current)
                    continue
            except (OSError, ValueError, KeyError):
                pass
        selections.append(tune_estimator(condition_cfg, method, show_progress))
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
    mode = settings.get('condition_mode', 'sweeps')
    if mode == 'selected':
        conditions = [(int(L), float(snr), 'condition') for L, snr in settings['conditions']]
    elif mode == 'grid':  # Read historical definitions; the GUI never offers a Cartesian grid.
        conditions = [(int(L), float(snr), 'condition') for L in settings['pilot_lengths']
                      for snr in settings['snr_values']]
    else:
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


def legacy_evaluation_definition(cfg, method, selected=None):
    spec = ESTIMATORS[method]
    definition = {'method': spec.name, 'revision': 1, 'nmse_scope': NMSE_SCOPE,
                  'communication': communication_definition(cfg, 'evaluation'),
                  'algorithms': legacy_solver_config(cfg, method), 'conditions': [list(x) for x in evaluation_conditions(cfg)],
                  'samples': cfg['run']['num_evaluation_samples'],
                  'archive_sha256': archive_identity(task_dir(cfg['run']['output_dir'], 'communication') / 'evaluation',
                      cfg['run']['num_evaluation_samples'], ['channels.npz', 'pilots.npz', 'observations.npz'])}
    if spec.tuning_grid:
        if tuning_settings(cfg, method)['selection_mode'] == 'per_condition':
            definition['tuning'] = legacy_tuning_definition(cfg, method)
            definition['selection_mode'] = 'per_condition'
        else:
            definition['selection'] = selected if selected is not None else load_selection(cfg, method)
    if spec.spatial:
        definition['spatial_dependence'] = spatial_dependence_definition(cfg, 'evaluation')
        definition['spatial_archive_sha256'] = archive_identity(task_dir(cfg['run']['output_dir'], 'spatial_dependence') / 'evaluation', cfg['run']['num_evaluation_samples'], ['matrix'])
    return definition


def evaluation_definition(cfg, method, selected=None, condition=None):
    from cgap_sbl_jadce.compatibility import solver_revision
    from cgap_sbl_jadce.communication_dataset import communication_condition_identity
    pairs = [tuple(condition)] if condition is not None else list(dict.fromkeys(value[:2] for value in evaluation_conditions(cfg)))
    definition = {'method': ESTIMATORS[method].name, 'revision': 2, 'nmse_scope': NMSE_SCOPE,
        'solver': solver_settings(cfg, method), 'code': solver_revision(method),
        'samples': cfg['run']['num_evaluation_samples'],
        'communication': [communication_condition_identity(cfg, 'evaluation', pair) for pair in pairs]}
    if ESTIMATORS[method].tuning_grid:
        definition['selections'] = [load_selection(cfg, method, pair) for pair in pairs]
    if ESTIMATORS[method].spatial:
        definition['spatial_dependence'] = spatial_dependence_definition(cfg, 'evaluation')
        definition['spatial_archive_sha256'] = archive_identity(task_dir(cfg['run']['output_dir'], 'spatial_dependence') / 'evaluation', cfg['run']['num_evaluation_samples'], ['matrix'])
    return definition


def evaluation_matches(saved, cfg, method, pair, expected=None):
    if saved.get('revision') == 1:
        from cgap_sbl_jadce.compatibility import legacy_definition_matches
        return legacy_definition_matches(saved, expected or legacy_evaluation_definition(cfg, method), cfg, method, pair)
    return saved == evaluation_definition(cfg, method, condition=pair)


def _aggregate(df):
    keys = ['algorithm', 'L', 'snr_db', 'sweep']
    aggregate = df.groupby(keys, as_index=False)[['precision', 'recall', 'f1', 'nmse', 'runtime_s']].mean()
    aggregate['nmse_scope'] = NMSE_SCOPE
    runtime = df.groupby(keys)['runtime_s'].agg(
        samples='count', mean_s='mean', std_s='std', median_s='median',
        min_s='min', max_s='max', total_s='sum', p95_s=lambda x: x.quantile(.95)).reset_index()
    return aggregate, runtime


@checkpointed
def run_estimator_stage(cfg, method, show_progress=True, conditions=None, resume=False):
    """Each L/SNR condition owns final data and its unfinished numerical units."""
    from cgap_sbl_jadce.condition_results import (completed_conditions, rebuild_algorithm_results,
                                                 migrate_legacy_results)
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
    definition = None
    migrate_legacy_results(cfg, method, definition, selections)
    save_task_snapshot(cfg, spec.task, status='started', action='run', method=spec.name)
    count = int(cfg['run']['num_evaluation_samples'])
    gamma_count = min(int(cfg['evaluation'].get('gamma_distribution_samples', 0)), count)
    pairs = list(dict.fromkeys(value[:2] for value in requested))
    done = {pair for pair, _, _ in completed_conditions(cfg, method, definition, selections)}
    for pair in tqdm(pairs, desc=f'{spec.name} conditions', unit='condition', disable=not show_progress):
        if resume and pair in done:
            continue
        L, snr = pair
        definition = evaluation_definition(cfg, method, condition=pair)
        folder = run_path(cfg, method, condition=pair)
        parameters = selections[pair]['parameters'] if spec.tuning_grid else {}
        signature = selection_digest(selections[pair]) if pair in selections else None
        checkpoint = Checkpoint(folder, {'definition': definition, 'condition': list(pair),
                                         'selection_digest': signature, 'gamma_count': gamma_count})
        save_json(folder / 'summary.json', {'status': 'running', 'condition': list(pair), 'definition': definition})
        rows, gammas, masks = [], [], []
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
                support = (support_from_gamma(result.gamma, parameters['tau']) if spec.tuning_grid
                           else support_from_rows(estimate, solver_parameters['B_q']))
                precision, recall, f1 = precision_recall_f1(r.a, support)
                row = {'q': q, 'algorithm': spec.name, 'S_q': S_q, 'L': L, 'snr_db': snr,
                       'precision': precision, 'recall': recall, 'f1': f1,
                       'nmse': nmse(r.Z, estimate, r.a), 'nmse_scope': NMSE_SCOPE,
                       'iterations': result.iterations if spec.tuning_grid else (solver_parameters['B_q'] if method == 'omp' else np.nan),
                       'runtime_s': runtime}
                if not spec.tuning_grid:
                    row.update(B_q=solver_parameters['B_q'], oracle_capped=solver_parameters['B_q'] < S_q)
                arrays = {'a': r.a, 'gamma': result.gamma} if spec.tuning_grid and q < gamma_count else {}
                checkpoint.save(f'{q:04d}', {'row': row}, **arrays)
            rows.extend({**row, 'sweep': view} for view in views)
            if arrays:
                masks.append(arrays['a']); gammas.append(arrays['gamma'])
        save_csv(folder / 'per_sample.csv', rows)
        aggregate, runtime = _aggregate(pd.DataFrame(rows))
        save_csv(folder / 'aggregate.csv', aggregate)
        save_csv(folder / 'runtime_summary.csv', runtime)
        files = [folder / name for name in ('per_sample.csv', 'aggregate.csv', 'runtime_summary.csv')]
        if gammas:
            save_npz(folder / 'gamma_distribution.npz', a=np.stack(masks), gamma=np.stack(gammas))
            files.append(folder / 'gamma_distribution.npz')
        files.extend(plot_condition_scores(cfg, method, pair))
        save_json(folder / 'summary.json', {
            'status': 'complete', 'condition': list(pair), 'definition': definition,
            'parameters': parameters, 'selection_digest': signature, 'Q': count,
            'gamma_samples': len(gammas), 'nmse_scope': NMSE_SCOPE,
            'outputs': {str(path.relative_to(folder)): digest
                        for path, digest in zip(files, output_hashes(files).values())},
        })
        # The lifecycle owner clears this condition only after data and figures commit.
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
    """Methods with complete, current scores for the requested condition subset."""
    from cgap_sbl_jadce.condition_results import completed_conditions
    requested = {value[:2] for value in evaluation_conditions(cfg, conditions)}
    available = []
    for method in ESTIMATORS:
        try:
            if run_path(cfg, method).exists():
                current = {pair for pair, _, _ in completed_conditions(cfg, method)}
                if requested <= current:
                    available.append(method)
            else:
                summary = json.loads(result_path(cfg['run']['output_dir'], method, 'run_summary.json').read_text())
                if (summary.get('status') in {'complete', 'partial'} and summary.get('nmse_scope') == NMSE_SCOPE and
                        all(evaluation_matches(summary.get('definition', {}), cfg, method, pair) for pair in requested)):
                    available.append(method)
        except (OSError, ValueError, KeyError):
            continue
    return available


def convergence_methods(cfg, condition=None):
    """Only completed, compatible SBL-family evaluation results are eligible."""
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
    """Summarize any selected subset of compatible completed algorithm outputs."""
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
            raise ValueError(f'Incompatible or historical {spec.name} results. Rerun {spec.name}.')
        if run_path(cfg, method).exists():
            from cgap_sbl_jadce.condition_results import completed_conditions
            df = pd.concat([frame for pair, _, frame in completed_conditions(cfg, method) if pair in {value[:2] for value in requested}], ignore_index=True)
        else:
            df = pd.read_csv(result_path(out, method, 'per_sample.csv'))
        expected = {(q, L, snr, sweep) for q in range(cfg['run']['num_evaluation_samples'])
                    for L, snr, sweep in requested}
        df = df[df.apply(lambda row: (row.L, row.snr_db, row.sweep) in requested, axis=1)]
        actual = set(df[['q', 'L', 'snr_db', 'sweep']].itertuples(index=False, name=None))
        if (len(df) != len(expected) or actual != expected or not df['algorithm'].eq(spec.name).all()
                or 'nmse_scope' not in df or not df['nmse_scope'].eq(NMSE_SCOPE).all()):
            raise ValueError(f'Incomplete or incompatible {spec.name} scores. Rerun {spec.name}.')
        datasets.append(df)
    df = pd.concat(datasets, ignore_index=True)
    folder = task_dir(out, 'comparison')
    folder.mkdir(parents=True, exist_ok=True)
    aggregate, runtime = _aggregate(df)
    save_csv(folder / 'per_sample.csv', df)
    save_csv(folder / 'aggregate.csv', aggregate)
    save_csv(folder / 'runtime_summary.csv', runtime)
    summary = {'status': 'complete', 'nmse_scope': NMSE_SCOPE,
               'conditions': [list(condition) for condition in requested], 'methods': methods,
               'algorithms': [ESTIMATORS[method].name for method in methods], 'metric_rows': len(df),
               'definitions': {method: [evaluation_definition(cfg, method, condition=pair) for pair in dict.fromkeys(value[:2] for value in requested)] for method in methods}}
    save_json(folder / 'summary.json', summary)
    save_task_snapshot(cfg, 'comparison', status='complete')
    print_task_report('ALGORITHM COMPARISON', {key: value for key, value in summary.items() if key != 'definitions'}, [folder])
    return summary


@checkpointed
def convergence_stage(cfg, show_progress=True, methods=None, realization_index=None, condition=None):
    """Record iteration histories for any selected available algorithms."""
    out = cfg['run']['output_dir']
    q = int(cfg['evaluation'].get('convergence_sample_offset', 0) if realization_index is None else realization_index)
    if not 0 <= q < cfg['run']['num_evaluation_samples']:
        raise ValueError('Convergence realization index is outside the evaluation dataset.')
    L, snr = condition or (int(cfg['evaluation']['pilot_length']), float(evaluation_settings(cfg)['snr_db']))
    r = load_communication_sample(validate_communication_dataset(cfg, 'evaluation', [(L, snr)]), q, (L, snr))[(L, snr)]
    methods = analysis_methods(cfg, 'convergence', methods, condition=(L, snr))
    if not set(methods) <= set(convergence_methods(cfg, (L, snr))):
        raise ValueError('Convergence requires completed SBL or CGAP-SBL-ANC evaluation at this condition.')
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
