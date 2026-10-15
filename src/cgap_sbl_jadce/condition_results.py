import json
import hashlib
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from cgap_sbl_jadce.file_cache import file_version, snapshot_cached
from cgap_sbl_jadce.io import save_csv, save_json, save_npz
from cgap_sbl_jadce.paths import result_path, run_path
from cgap_sbl_jadce.checkpoints import output_hashes
from cgap_sbl_jadce.metrics import NMSE_SCOPE
from cgap_sbl_jadce.settings import evaluation_settings


def pending_reset_pairs(cfg, method):
    from cgap_sbl_jadce.experiments.estimators import evaluation_conditions
    folder = run_path(cfg, method)
    pairs = set()
    for path in folder.glob('.resets/[0-9a-f]*.json'):
        try:
            pairs.update(tuple(pair) for pair in json.loads(path.read_text())['conditions'])
        except FileNotFoundError:
            continue
    for path in (folder.parent / '.recovery').glob('run*.json'):
        try:
            saved = json.loads(path.read_text())
        except FileNotFoundError:
            continue
        if saved.get('phase') == 'reset_pending' and not saved.get('superseded'):
            selected = saved['request'].get('conditions')
            if selected is None:
                pairs.update(value[:2] for value in evaluation_conditions(cfg))
            else:
                pairs.update(tuple(pair) for pair in selected)
    return pairs


def reset_run_results(cfg, method, pairs, definitions, resume):
    from cgap_sbl_jadce.checkpoints import scientific_revision
    from cgap_sbl_jadce.compatibility import require_match
    from cgap_sbl_jadce.experiments.estimators import evaluation_conditions, ESTIMATORS
    allowed = {value[:2] for value in evaluation_conditions(cfg, pairs, method=method)}
    if set(pairs) != allowed or len(pairs) != len(definitions):
        raise ValueError('Reset scope must match the requested evaluation conditions.')
    folder = run_path(cfg, method)
    if folder.is_symlink() or folder.parent.is_symlink():
        raise ValueError('Cannot reset results through a linked algorithm folder.')
    scope = hashlib.sha256(json.dumps(sorted(pairs)).encode()).hexdigest()[:16]
    marker = folder / '.resets' / f'{scope}.json'
    if marker.parent.is_symlink():
        raise ValueError('Cannot reset results through a linked reset folder.')
    recoveries = []
    job_id = os.environ.get('CGAP_JOB_DIRECTORY', '').rstrip('/').rsplit('/', 1)[-1]
    for path in (folder.parent / '.recovery').glob('run*.json'):
        saved = json.loads(path.read_text())
        selected = saved['request'].get('conditions')
        scope_pairs = {tuple(pair) for pair in selected} if selected is not None else {value[:2] for value in evaluation_conditions(cfg)}
        same_scope = scope_pairs == set(pairs)
        owned = not saved.get('superseded') and (
            saved.get('job_id', '') == job_id and (bool(job_id) or same_scope) or resume and same_scope)
        recoveries.append((path, saved, scope_pairs, owned))
    replaced_markers = []
    for path in folder.glob('.resets/[0-9a-f]*.json'):
        if path == marker:
            continue
        previous = {tuple(pair) for pair in json.loads(path.read_text())['conditions']}
        if previous & set(pairs):
            if not previous <= set(pairs) or (resume and not marker.exists()):
                raise ValueError('Finish the interrupted reset or explicitly rerun its entire condition scope.')
            replaced_markers.append(path)
    for _, saved, scope_pairs, owned in recoveries:
        if (saved.get('phase') == 'reset_pending' and not saved.get('superseded') and not owned
                and scope_pairs & set(pairs) and not scope_pairs <= set(pairs)):
            raise ValueError('Finish the interrupted reset or explicitly rerun its entire condition scope.')
    if resume and not marker.exists() and not any(saved.get('phase') == 'reset_pending' and owned for _, saved, _, owned in recoveries):
        return
    identity = {'conditions': [list(pair) for pair in pairs], 'definitions': definitions, 'code': scientific_revision()}
    if resume and marker.exists():
        require_match(json.loads(marker.read_text()), identity, 'Pending run reset')
    targets = []
    for pair in pairs:
        target = run_path(cfg, method, condition=pair)
        if target.is_symlink() or target.resolve().parent != folder.resolve():
            raise ValueError('Reset target is outside the selected algorithm run folder.')
        files = [target / name for name in ('summary.json', 'per_sample.csv', 'aggregate.csv',
                                           'runtime_summary.csv', 'gamma_distribution.npz')]
        files.extend(target / 'figures' / f'realization_scores.{suffix}' for suffix in ('png', 'pdf'))
        directories = []
        for metadata in target.glob('.checkpoints/*/definition.json'):
            saved = json.loads(metadata.read_text())['definition']
            if (saved.get('definition', {}).get('method') != ESTIMATORS[method].name or
                    tuple(saved.get('condition', ())) != pair):
                raise ValueError('Checkpoint ownership differs from the selected condition.')
            directories.append(metadata.parent)
            files.extend([metadata, *metadata.parent.glob('*.npz')])
        if any(not path.resolve().is_relative_to(target.resolve()) for path in files):
            raise ValueError('A selected result file points outside its condition folder.')
        targets.append((target, files, directories))
    save_json(marker, identity)
    for path in replaced_markers:
        path.unlink()
    print(f'Resetting {len(pairs)} {method} conditions.', flush=True)
    for _, files, directories in targets:
        for path in files:
            path.unlink(missing_ok=True)
        for directory in directories:
            if not any(directory.iterdir()):
                directory.rmdir()
    for path, saved, scope_pairs, owned in recoveries:
        if owned:
            continue
        if scope_pairs & set(pairs):
            save_json(path, dict(saved, superseded=True))
    rebuild_algorithm_results(cfg, method)
    for path, saved, _, owned in recoveries:
        if owned:
            save_json(path, dict(saved, phase='calculating'))
    marker.unlink()
    if not any(marker.parent.iterdir()):
        marker.parent.rmdir()


@lru_cache(maxsize=128)
def _condition_table(path, size, modified, changed):
    return pd.read_csv(path)


@snapshot_cached
def completed_conditions(cfg, method, selections=None):
    from cgap_sbl_jadce.experiments.estimators import (evaluation_definition, evaluation_conditions,
                                                     evaluation_selections, selection_digest, evaluation_matches)
    requested = evaluation_conditions(cfg, method=method, include_optional=True)
    resetting = pending_reset_pairs(cfg, method)
    if selections is None:
        selections = {}
        for pair in dict.fromkeys(value[:2] for value in requested):
            try:
                selections.update(evaluation_selections(cfg, method, evaluation_conditions(cfg, [pair], method=method)))
            except (OSError, ValueError):
                pass
    result = []
    for pair in dict.fromkeys(value[:2] for value in requested):
        if pair in resetting:
            continue
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
            frame = _condition_table(*file_version(folder / 'per_sample.csv')).copy()
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
    if not gamma_count:
        result_path(cfg['run']['output_dir'], method, 'gamma_distribution.npz').unlink(missing_ok=True)
    if rows:
        save_csv(result_path(cfg['run']['output_dir'], method, 'per_sample.csv'), rows)
        aggregate, runtime = _aggregate(pd.DataFrame(rows))
        save_csv(result_path(cfg['run']['output_dir'], method, 'aggregate.csv'), aggregate)
        save_csv(result_path(cfg['run']['output_dir'], method, 'runtime_summary.csv'), runtime)
    else:
        for name in ('per_sample.csv', 'aggregate.csv', 'runtime_summary.csv'):
            result_path(cfg['run']['output_dir'], method, name).unlink(missing_ok=True)
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
