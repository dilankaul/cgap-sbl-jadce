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
from cgap_sbl_jadce.paths import result_path, task_dir, tuning_path
from cgap_sbl_jadce.provenance import save_task_snapshot
from cgap_sbl_jadce.reporting import print_task_report
from cgap_sbl_jadce.checkpoints import Checkpoint, archive_identity


def _sbl_options(cfg):
    return {key: cfg['algorithms'][key] for key in ('gamma_init', 'K', 'delta')}


def _omp(r, C, parameters, cfg, **kwargs):
    return mmv_omp(r.Theta, r.Y, parameters['B_q'])


def _cosamp(r, C, parameters, cfg, **kwargs):
    return mmv_cosamp(r.Theta, r.Y, parameters['B_q'],
                      max_iter=cfg['algorithms']['cosamp_max_iter'], tol=cfg['algorithms']['cosamp_tol'])


def _sbl(r, C, parameters, cfg, **kwargs):
    return sbl(r.Theta, r.Y, r.sigma2, **_sbl_options(cfg), **kwargs)


def _anc(r, C, parameters, cfg, **kwargs):
    return cgap_sbl_anc(r.Theta, r.Y, r.sigma2, C,
                        parameters['alpha'], parameters['beta'], **_sbl_options(cfg), **kwargs)


def _anc_grid(cfg):
    return [{'alpha': float(alpha), 'beta': float(beta)}
            for alpha in cfg['tuning']['alpha_values'] for beta in cfg['tuning']['beta_values']]


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


def tuning_definition(cfg, method):
    spec = ESTIMATORS[method]
    definition = {'method': spec.name, 'revision': 1, 'nmse_scope': NMSE_SCOPE,
                  'condition': [cfg['tuning']['reference_pilot_length'], cfg['tuning']['reference_snr_db']],
                  'tau_grid': {'start': 0., 'stop': .2, 'step': .001},
                  'algorithms': cfg['algorithms'],
                  'communication': communication_definition(cfg, 'tuning'),
                  'parameters': spec.tuning_grid(cfg), 'samples': cfg['run']['num_tuning_samples'],
                  'archive_sha256': archive_identity(task_dir(cfg['run']['output_dir'], 'communication') / 'tuning',
                      cfg['run']['num_tuning_samples'], ['channels.npz', 'pilots.npz', 'observations.npz'])}
    if spec.spatial:
        definition['spatial_dependence'] = spatial_dependence_definition(cfg, 'tuning')
        definition['spatial_archive_sha256'] = archive_identity(task_dir(cfg['run']['output_dir'], 'spatial_dependence') / 'tuning', cfg['run']['num_tuning_samples'], ['matrix'])
    return definition


def condition_config(cfg, condition):
    cfg = deepcopy(cfg)
    cfg['tuning']['reference_pilot_length'], cfg['tuning']['reference_snr_db'] = condition
    return cfg


def tuning_conditions(cfg, requested=None):
    pairs = [tuple(pair) for pair in communication_definition(cfg, 'tuning')['conditions']]
    if cfg['tuning'].get('selection_mode', 'reference') != 'per_condition':
        reference = (cfg['tuning']['reference_pilot_length'], cfg['tuning']['reference_snr_db'])
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
    if condition is not None and cfg['tuning'].get('selection_mode', 'reference') == 'per_condition':
        cfg = condition_config(cfg, condition)
    path = tuning_path(cfg, method, 'selected.json')
    spec = ESTIMATORS[method]
    if not path.exists():
        raise FileNotFoundError(f'Choose Algorithms > {spec.name} > Tune : {path}')
    selected = json.loads(path.read_text())
    definition = deepcopy(expected) if expected is not None else tuning_definition(cfg, method)
    definition['condition'] = [cfg['tuning']['reference_pilot_length'], cfg['tuning']['reference_snr_db']]
    if selected.get('definition') != definition:
        raise ValueError(f'Saved tuning differs from the method, NMSE definition or setup. Choose Tune for {spec.name}.')
    return selected


def selection_digest(selected):
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def evaluation_selections(cfg, method, required):
    spec = ESTIMATORS[method]
    if not spec.tuning_grid:
        return {}
    expected = tuning_definition(cfg, method)
    selections = {}
    for pair in dict.fromkeys((L, snr) for L, snr, _ in evaluation_conditions(cfg)):
        try:
            selections[pair] = load_selection(cfg, method, pair, expected=expected)
        except (OSError, ValueError):
            if pair in {(L, snr) for L, snr, _ in required}:
                raise
    return selections


def tune_estimator(cfg, method, show_progress=True):
    spec = ESTIMATORS[method]
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
        curve = threshold_curve(np.stack([r.a for r in communication]), np.asarray(gammas))
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
                'reference_pilot_length': cfg['tuning']['reference_pilot_length'],
                'reference_snr_db': cfg['tuning']['reference_snr_db']}
    out = cfg['run']['output_dir']
    save_json(tuning_path(cfg, method, 'selected.json'), selected)
    save_csv(tuning_path(cfg, method, 'thresholds.csv'), curves[winner])
    if spec.spatial:
        save_csv(tuning_path(cfg, method, 'alpha_beta.csv'), [rows[i] for i in order])
        save_csv(tuning_path(cfg, method, 'alpha_beta_tau_scores.csv'),
                 [{**parameters, **point} for parameters, curve in zip(spec.tuning_grid(cfg), curves)
                  for point in curve])
    save_json(tuning_path(cfg, method, 'summary.json'),
              {'selected': selected, 'grid_pairs': len(rows), 'tau_candidates': 201})
    return selected



def tune_estimator_stage(cfg, method, show_progress=True, conditions=None):
    """Tune one algorithm using only tuning realizations; save its selection."""
    spec = ESTIMATORS[method]
    if spec.tuning_grid is None:
        raise ValueError(f"{spec.name} does not require tuning.")
    save_task_snapshot(cfg, spec.task, status='started', action='tune', method=spec.name)
    pairs = tuning_conditions(cfg, conditions)
    selections = [tune_estimator(condition_config(cfg, pair), method, show_progress) for pair in pairs]
    selected = selections[0] if len(selections) == 1 else {'conditions': selections}
    save_task_snapshot(cfg, spec.task, status='complete', action='tune',
                       method=spec.name, selected_parameters=selected)
    print_task_report(f'{spec.name} TUNING',
                      {'tuned conditions': len(selections), 'selections': [item['parameters'] for item in selections]},
                      [tuning_path(condition_config(cfg, pair), method, 'selected.json') for pair in pairs])
    return selected


def evaluation_conditions(cfg, selected=None):
    L0 = int(cfg['evaluation']['pilot_length'])
    snr0 = float(cfg['tuning']['reference_snr_db'])
    if cfg['evaluation'].get('condition_mode', 'sweeps') == 'grid':
        conditions = [(int(L), float(snr), 'condition') for L in cfg['system']['pilot_lengths']
                      for snr in cfg['system']['snr_db']]
    else:
        conditions = ([(L0, float(snr), 'snr_sweep') for snr in cfg['system']['snr_db']]
                      + [(int(L), snr0, 'pilot_sweep') for L in cfg['system']['pilot_lengths']])
    if selected is None:
        return conditions
    pairs = {tuple(pair) for pair in selected}
    if not pairs or not pairs <= {(L, snr) for L, snr, _ in conditions}:
        raise ValueError('Choose at least one L/SNR pair from the setup evaluation conditions.')
    return [condition for condition in conditions if condition[:2] in pairs]


def evaluation_definition(cfg, method, selected=None):
    spec = ESTIMATORS[method]
    definition = {'method': spec.name, 'revision': 1, 'nmse_scope': NMSE_SCOPE,
                  'communication': communication_definition(cfg, 'evaluation'),
                  'algorithms': cfg['algorithms'], 'conditions': [list(x) for x in evaluation_conditions(cfg)],
                  'samples': cfg['run']['num_evaluation_samples'],
                  'archive_sha256': archive_identity(task_dir(cfg['run']['output_dir'], 'communication') / 'evaluation',
                      cfg['run']['num_evaluation_samples'], ['channels.npz', 'pilots.npz', 'observations.npz'])}
    if spec.tuning_grid:
        if cfg['tuning'].get('selection_mode', 'reference') == 'per_condition':
            definition['tuning'] = tuning_definition(cfg, method)
            definition['selection_mode'] = 'per_condition'
        else:
            definition['selection'] = selected if selected is not None else load_selection(cfg, method)
    if spec.spatial:
        definition['spatial_dependence'] = spatial_dependence_definition(cfg, 'evaluation')
        definition['spatial_archive_sha256'] = archive_identity(task_dir(cfg['run']['output_dir'], 'spatial_dependence') / 'evaluation', cfg['run']['num_evaluation_samples'], ['matrix'])
    return definition


def _aggregate(df):
    keys = ['algorithm', 'L', 'snr_db', 'sweep']
    aggregate = df.groupby(keys, as_index=False)[['precision', 'recall', 'f1', 'nmse', 'runtime_s']].mean()
    aggregate['nmse_scope'] = NMSE_SCOPE
    runtime = df.groupby(keys)['runtime_s'].agg(
        samples='count', mean_s='mean', std_s='std', median_s='median',
        min_s='min', max_s='max', total_s='sum', p95_s=lambda x: x.quantile(.95)).reset_index()
    return aggregate, runtime


def run_estimator_stage(cfg, method, show_progress=True, conditions=None):
    """Evaluate one algorithm using its saved selection; never retune it."""
    spec = ESTIMATORS[method]
    out = cfg['run']['output_dir']
    communication_folder = validate_communication_dataset(cfg, 'evaluation')
    C_folder = validate_spatial_dependence_dataset(cfg, 'evaluation') if spec.spatial else None
    requested = evaluation_conditions(cfg, conditions)
    selections = evaluation_selections(cfg, method, requested)
    selected = next(iter(selections.values()), None)
    definition = evaluation_definition(cfg, method, selected)
    save_task_snapshot(cfg, spec.task, status='started', action='run', method=spec.name)
    summary_path = result_path(out, method, 'run_summary.json')
    save_json(summary_path, {'status': 'started', 'method': spec.name})
    parameters = selected['parameters'] if selected else {}
    signatures = {f'{L}_{snr:g}': selection_digest(item) for (L, snr), item in selections.items()}
    rows, gammas, masks = [], [], []
    count = int(cfg['run']['num_evaluation_samples'])
    reference = (int(cfg['evaluation']['pilot_length']), float(cfg['tuning']['reference_snr_db']))
    gamma_count = min(int(cfg['evaluation'].get('gamma_distribution_samples', 0)), count)
    checkpoint = Checkpoint(task_dir(out, spec.task) / 'evaluation', {'definition': definition, 'gamma_count': gamma_count})
    all_conditions = evaluation_conditions(cfg)
    for q in tqdm(range(count), desc=f'{spec.name} evaluation',
                  unit='realization', disable=not show_progress):
        cached = checkpoint.load(f'{q:04d}')
        sample_rows = cached[0]['rows'] if cached else []
        gamma_data = cached[1] if cached else {}
        if spec.tuning_grid and cached:
            sample_rows = [row for row in sample_rows if
                cached[0].get('selections', {}).get(f"{row['L']}_{row['snr_db']:g}") == signatures.get(f"{row['L']}_{row['snr_db']:g}")
                and (row['L'], row['snr_db']) in selections]
            if not any((row['L'], row['snr_db']) == reference for row in sample_rows):
                gamma_data = {}
        done = {(row['L'], row['snr_db'], row['sweep']) for row in sample_rows}
        if all(condition in done for condition in requested):
            rows.extend(sample_rows)
            if gamma_data:
                masks.append(gamma_data['a']); gammas.append(gamma_data['gamma'])
            continue
        communication = load_communication_sample(communication_folder, q)
        C = load_spatial_dependence_matrix(C_folder, q) if C_folder else None
        for L, snr, sweep in requested:
            if (L, snr, sweep) in done:
                continue
            # The reference pair can appear in both sweep plots. It represents
            # one calculation; keep both views with the same measured runtime.
            previous = next((row for row in sample_rows if (row['L'], row['snr_db']) == (L, snr)), None)
            if previous is not None:
                sample_rows.append({**previous, 'sweep': sweep})
                checkpoint.save(f'{q:04d}', {'rows': sample_rows, 'selections': signatures}, **gamma_data)
                continue
            r = communication[(L, snr)]
            S_q = int(r.a.sum())
            parameters = selections[(L, snr)]['parameters'] if spec.tuning_grid else {}
            solver_parameters = parameters if spec.tuning_grid else {'B_q': oracle_sparsity_budget(S_q, L, len(r.a))}
            start = time.perf_counter()
            result = spec.solve(r, C, solver_parameters, cfg)
            runtime = time.perf_counter() - start
            estimate = result.mu if spec.tuning_grid else result
            support = (support_from_gamma(result.gamma, parameters['tau']) if spec.tuning_grid
                       else support_from_rows(estimate, solver_parameters['B_q']))
            precision, recall, f1 = precision_recall_f1(r.a, support)
            row = {'q': q, 'algorithm': spec.name, 'S_q': S_q, 'L': L, 'snr_db': snr, 'sweep': sweep,
                   'precision': precision, 'recall': recall, 'f1': f1,
                   'nmse': nmse(r.Z, estimate, r.a), 'nmse_scope': NMSE_SCOPE,
                   'iterations': result.iterations if spec.tuning_grid else (solver_parameters['B_q'] if method == 'omp' else np.nan),
                   'runtime_s': runtime}
            if not spec.tuning_grid:
                row.update(B_q=solver_parameters['B_q'], oracle_capped=solver_parameters['B_q'] < S_q)
            sample_rows.append(row)
            if spec.tuning_grid and q < gamma_count and (L, snr) == reference :
                gamma_data = {'a': r.a, 'gamma': result.gamma}
            checkpoint.save(f'{q:04d}', {'rows': sample_rows, 'selections': signatures}, **gamma_data)
        rows.extend(sample_rows)
        if gamma_data:
            masks.append(gamma_data['a']); gammas.append(gamma_data['gamma'])
    raw = result_path(out, method, 'per_sample.csv')
    save_csv(raw, rows)
    if gammas:
        save_npz(result_path(out, method, 'gamma_distribution.npz'), a=np.stack(masks), gamma=np.stack(gammas))
    df = pd.DataFrame(rows)
    completed = [condition for condition in all_conditions
                 if len(df[(df.L == condition[0]) & (df.snr_db == condition[1]) & (df.sweep == condition[2])]) == count]
    complete_rows = df.set_index(['L', 'snr_db', 'sweep']).loc[completed].reset_index()
    aggregate, runtime = _aggregate(complete_rows)
    save_csv(result_path(out, method, 'aggregate.csv'), aggregate)
    save_csv(result_path(out, method, 'runtime_summary.csv'), runtime)
    summary = {'status': 'complete' if len(completed) == len(all_conditions) else 'partial',
               'completed_conditions': [list(condition) for condition in completed],
               'requested_conditions': [list(condition) for condition in requested],
               'selection_digests': signatures, 'definition': definition, 'Q': count,
               'metric_rows': len(rows), 'solver_units': len({(row['q'], row['L'], row['snr_db']) for row in rows}),
               'nmse_scope': NMSE_SCOPE, 'gamma_samples': len(gammas),
               'reference_pilot_length': reference[0], 'reference_snr_db': reference[1],
               'oracle_capped_conditions': sum(bool(row.get('oracle_capped', False)) for row in rows),
               'timing': {'clock': 'time.perf_counter', 'unit': 'seconds', 'scope': 'one solver call',
                          'includes': 'initialization, solver validation, all iterations, final channel estimate',
                          'excludes': 'tuning, data/matrix generation, loading, thresholding, metrics, plotting, saving'}}
    save_json(summary_path, summary)
    save_task_snapshot(cfg, spec.task, status='complete', action='run', method=spec.name, selected_parameters=selected)
    print_task_report(f'{spec.name}',
                      {'evaluation realizations': count, 'metric rows': len(rows),
                       'NMSE scope': NMSE_SCOPE, 'selected parameters': parameters},
                      [raw, summary_path, task_dir(out, spec.task)])
    return summary


def completed_methods(cfg, conditions=None):
    """Methods with complete, current scores for the requested condition subset."""
    requested = evaluation_conditions(cfg, conditions)
    available = []
    for method in ESTIMATORS:
        path = result_path(cfg['run']['output_dir'], method, 'run_summary.json')
        if not path.exists():
            continue
        try:
            summary = json.loads(path.read_text())
            done = {tuple(value) for value in summary.get('completed_conditions', [])}
            if summary.get('status') == 'complete' and not done:
                done = set(evaluation_conditions(cfg))
            if not (summary.get('status') in {'complete','partial'} and summary.get('nmse_scope') == NMSE_SCOPE
                    and set(requested) <= done and summary.get('definition') == evaluation_definition(cfg, method)):
                continue
            if ESTIMATORS[method].tuning_grid:
                selections = evaluation_selections(cfg, method, requested)
                if any(summary.get('selection_digests', {}).get(f'{L}_{snr:g}') != selection_digest(item)
                       for (L,snr),item in selections.items()):
                    continue
            if not result_path(cfg['run']['output_dir'], method, 'per_sample.csv').exists():
                continue
            available.append(method)
        except (OSError, ValueError, KeyError):
            continue
    return available


def convergence_methods(cfg, condition=None):
    """Only completed, compatible SBL-family evaluation results are eligible."""
    condition = condition or (cfg['evaluation']['pilot_length'], cfg['tuning']['reference_snr_db'])
    try:
        completed = completed_methods(cfg, [condition])
        validate_communication_dataset(cfg, 'evaluation')
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
        if summary.get('status') not in {'complete', 'partial'} or summary.get('definition') != evaluation_definition(cfg, method):
            raise ValueError(f'Incompatible or historical {spec.name} results. Rerun {spec.name}.')
        if spec.tuning_grid and cfg['tuning'].get('selection_mode', 'reference') == 'per_condition':
            selections = evaluation_selections(cfg, method, requested)
            if any(summary.get('selection_digests', {}).get(f'{L}_{snr:g}') != selection_digest(item)
                   for (L, snr), item in selections.items() if (L, snr) in {value[:2] for value in requested}):
                raise ValueError(f'Saved {spec.name} scores use outdated selections. Rerun {spec.name}.')
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
               'algorithms': [ESTIMATORS[method].name for method in methods], 'metric_rows': len(df)}
    save_json(folder / 'summary.json', summary)
    save_task_snapshot(cfg, 'comparison', status='complete')
    print_task_report('ALGORITHM COMPARISON', summary, [folder])
    return summary


def convergence_stage(cfg, show_progress=True, methods=None, realization_index=None, condition=None):
    """Record iteration histories for any selected available algorithms."""
    out = cfg['run']['output_dir']
    q = int(cfg['evaluation'].get('convergence_sample_offset', 0) if realization_index is None else realization_index)
    if not 0 <= q < cfg['run']['num_evaluation_samples']:
        raise ValueError('Convergence realization index is outside the evaluation dataset.')
    L, snr = condition or (int(cfg['evaluation']['pilot_length']), float(cfg['tuning']['reference_snr_db']))
    r = load_communication_sample(validate_communication_dataset(cfg, 'evaluation'), q)[(L, snr)]
    methods = analysis_methods(cfg, 'convergence', methods, condition=(L, snr))
    if not set(methods) <= set(convergence_methods(cfg, (L, snr))):
        raise ValueError('Convergence requires completed SBL or CGAP-SBL-ANC evaluation at this condition.')
    C = load_spatial_dependence_matrix(validate_spatial_dependence_dataset(cfg, 'evaluation'), q) if any(ESTIMATORS[method].spatial for method in methods) else None
    histories, scores = {}, {}
    selections = {method: load_selection(cfg, method, (L, snr)) for method in methods if ESTIMATORS[method].tuning_grid}
    checkpoint = Checkpoint(task_dir(out, 'convergence'), {'q': q, 'L': L, 'snr': snr, 'selections': selections,
        'methods': {method: evaluation_definition(cfg, method) for method in methods}})
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
    summary = {'status': 'complete', 'q': q, 'L': L, 'snr_db': snr, 'methods': methods, 'nmse_scope': NMSE_SCOPE, 'scores': scores}
    save_json(result_path(out, 'convergence', 'summary.json'), summary)
    save_task_snapshot(cfg, 'convergence', status='complete', sample_index=q)
    print_task_report('CONVERGENCE', summary, [task_dir(out, 'convergence')])
    return summary
