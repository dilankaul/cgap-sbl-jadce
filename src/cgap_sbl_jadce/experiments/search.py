import hashlib
import json
import time

import numpy as np
from tqdm.auto import tqdm

from cgap_sbl_jadce.experiments.tuning import threshold_curve
from cgap_sbl_jadce.metrics import nmse


def candidate_key(parameters):
    return hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:24]


def rank_score(row):
    error = row['nmse']
    return (row['f1'], -error if np.isfinite(error) else -np.inf, row['recall'])


def score_candidate(cfg, parameters, communication, matrices, checkpoint, solve, tau_values,
                    count, show_progress=True, label='Tuning realizations', prefix=None):
    gammas, errors, iterations, runtimes = [], [], [], []
    prefix = prefix or 'candidate_' + candidate_key(parameters)
    for q in tqdm(range(count), desc=label, unit='realization', disable=not show_progress):
        key = f'{prefix}_{q:04d}'
        cached = checkpoint.load(key)
        if cached is None:
            started = time.perf_counter()
            result = solve(communication[q], matrices[q], parameters, cfg)
            elapsed = time.perf_counter() - started
            metadata = {'nmse': nmse(communication[q].Z, result.mu, communication[q].a),
                        'iterations': result.iterations, 'runtime_s': elapsed}
            arrays = {'gamma': result.gamma}
            checkpoint.save(key, metadata, **arrays)
        else:
            metadata, arrays = cached
        gammas.append(arrays['gamma'])
        errors.append(metadata['nmse'])
        iterations.append(metadata['iterations'])
        runtimes.append(metadata['runtime_s'])
    curve = threshold_curve(np.stack([r.a for r in communication[:count]]), np.asarray(gammas), tau_values)
    best = max(curve, key=lambda point: (point['f1'], point['recall'], point['precision']))
    finite = np.asarray(errors)[np.isfinite(errors)]
    return ({**parameters, **best, 'nmse': float(np.mean(finite)) if len(finite) else float('nan'),
             'iterations': float(np.mean(iterations)), 'solver_runtime_s': float(np.sum(runtimes)),
             'samples': count}, curve)
