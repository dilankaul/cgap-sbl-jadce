import json

import numpy as np
import pandas as pd

from cgap_sbl_jadce.io import save_csv, save_json, save_npz
from cgap_sbl_jadce.paths import result_path, run_path
from cgap_sbl_jadce.checkpoints import output_hashes
from cgap_sbl_jadce.metrics import NMSE_SCOPE
from cgap_sbl_jadce.settings import evaluation_settings


def completed_conditions(cfg, method, selections=None):
    from cgap_sbl_jadce.experiments.estimators import (evaluation_definition, evaluation_conditions,
                                                     evaluation_selections, selection_digest, evaluation_matches)
    requested = evaluation_conditions(cfg, method=method, include_optional=True)
    if selections is None:
        selections = {}
        for pair in dict.fromkeys(value[:2] for value in requested):
            try:
                selections.update(evaluation_selections(cfg, method, evaluation_conditions(cfg, [pair], method=method)))
            except (OSError, ValueError):
                pass
    result = []
    for pair in dict.fromkeys(value[:2] for value in requested):
        folder = run_path(cfg, method, condition=pair)
        try:
            summary = json.loads((folder / 'summary.json').read_text())
            if summary['status'] != 'complete' or not evaluation_matches(summary['definition'], cfg, method, pair):
                continue
            if summary.get('selection_digest') != (selection_digest(selections[pair]) if pair in selections else None):
                continue
            files = [folder / name for name in summary['outputs']]
            hashes = {str(path.relative_to(folder)): digest
                      for path, digest in zip(files, output_hashes(files).values())}
            if hashes != summary['outputs']:
                continue
            frame = pd.read_csv(folder / 'per_sample.csv')
            expected = set(range(cfg['run']['num_evaluation_samples']))
            actual = set(frame.q)
            if (actual != expected or not frame.L.eq(pair[0]).all() or not frame.snr_db.eq(pair[1]).all()
                    or not frame.nmse_scope.eq(NMSE_SCOPE).all() or
                    not frame.algorithm.eq(summary['definition']['method']).all()):
                continue
            unique = frame.drop(columns='sweep').drop_duplicates('q')
            if frame.drop(columns='sweep').drop_duplicates().shape[0] != len(expected):
                continue

            frame = pd.concat([unique.assign(sweep=view) for L, snr, view in requested if (L, snr) == pair], ignore_index=True)
            result.append((pair, summary, frame))
        except (OSError, ValueError, KeyError):
            continue
    return result


def rebuild_algorithm_results(cfg, method, selections=None):
    from cgap_sbl_jadce.experiments.estimators import (_aggregate, evaluation_definition,
                                                     evaluation_conditions, ESTIMATORS)
    definition = {"method": ESTIMATORS[method].name, "revision": 2}
    records = completed_conditions(cfg, method, selections)
    required_conditions = evaluation_conditions(cfg)
    all_conditions = evaluation_conditions(cfg, method=method, include_optional=True)
    order = {value: i for i, value in enumerate(all_conditions)}
    rows = sorted([row for _, _, frame in records for row in frame.to_dict('records')],
                  key=lambda row: (row['q'], order[(row['L'], row['snr_db'], row['sweep'])]))
    done = [value for value in all_conditions if value[:2] in {pair for pair, _, _ in records}]
    gamma_count = 0
    reference = (int(cfg['evaluation']['pilot_length']), float(evaluation_settings(cfg)['snr_db']))
    for pair, summary, _ in records:
        if pair == reference and summary.get('gamma_samples', 0):
            with np.load(run_path(cfg, method, 'gamma_distribution.npz', pair)) as archive:
                save_npz(result_path(cfg['run']['output_dir'], method, 'gamma_distribution.npz'),
                         a=archive['a'], gamma=archive['gamma'])
            gamma_count = summary['gamma_samples']
    if rows:
        save_csv(result_path(cfg['run']['output_dir'], method, 'per_sample.csv'), rows)
        aggregate, runtime = _aggregate(pd.DataFrame(rows))
        save_csv(result_path(cfg['run']['output_dir'], method, 'aggregate.csv'), aggregate)
        save_csv(result_path(cfg['run']['output_dir'], method, 'runtime_summary.csv'), runtime)
    summary = {
        'status': 'complete' if {value[:2] for value in required_conditions} <= {pair for pair, _, _ in records} else 'partial',
        'definition': definition, 'method': ESTIMATORS[method].name,
        'completed_conditions': [list(value) for value in done],
        'completed_pairs': [list(pair) for pair, _, _ in records],
        'total_conditions': len({value[:2] for value in required_conditions}),
        'optional_conditions': [list(value[:2]) for value in all_conditions if value[2] == 'condition' and value[:2] not in {item[:2] for item in required_conditions}],
        'Q': cfg['run']['num_evaluation_samples'], 'metric_rows': len(rows),
        'solver_units': len({(row['q'], row['L'], row['snr_db']) for row in rows}),
        'selection_digests': {f'{L}_{snr:g}': record.get('selection_digest')
                              for (L, snr), record, _ in records if record.get('selection_digest')},
        'nmse_scope': NMSE_SCOPE, 'gamma_samples': gamma_count,
        'reference_pilot_length': reference[0], 'reference_snr_db': reference[1],
        'oracle_capped_conditions': sum(bool(row.get('oracle_capped', False)) for row in rows),
        'timing': {'clock': 'time.perf_counter', 'unit': 'seconds', 'scope': 'one solver call',
                   'includes': 'initialization, solver validation, all iterations, final channel estimate',
                   'excludes': 'tuning, data/matrix generation, loading, thresholding, metrics, plotting, saving'},
    }
    save_json(result_path(cfg['run']['output_dir'], method, 'run_summary.json'), summary)
    return summary


def condition_issue(cfg, method, pair):
    from cgap_sbl_jadce.experiments.estimators import evaluation_definition, evaluation_matches, ESTIMATORS
    from cgap_sbl_jadce.compatibility import require_match
    try:
        summary = json.loads(run_path(cfg, method, 'summary.json', pair).read_text())
        saved = summary.get('definition', {})
        require_match(saved, evaluation_definition(cfg, method, condition=pair), ESTIMATORS[method].name + ' results')
    except (OSError, ValueError, KeyError) as error:
        return str(error)
    return 'A required result/figure is missing, incomplete or differs from its recorded hash.'
