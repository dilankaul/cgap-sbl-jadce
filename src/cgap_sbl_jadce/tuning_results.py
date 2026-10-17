import hashlib
import json
import os

from cgap_sbl_jadce.io import save_json
from cgap_sbl_jadce.paths import task_dir, tuning_path, tuning_figure_path


def pending_tuning_resets(cfg, method):
    from cgap_sbl_jadce.experiments.estimators import tuning_conditions
    root = task_dir(cfg['run']['output_dir'], method)
    pairs = set()
    for path in (root / 'tuning').glob('.resets/[0-9a-f]*.json'):
        pairs.update(tuple(pair) for pair in json.loads(path.read_text())['conditions'])
    for path in root.glob('.recovery/tune*.json'):
        saved = json.loads(path.read_text())
        if saved.get('phase') == 'reset_pending' and not saved.get('superseded'):
            pairs.update(tuning_conditions(cfg, saved['request'].get('conditions'), method))
    return pairs


def completed_tuning_selection(cfg, method, pair):
    from cgap_sbl_jadce.checkpoints import output_hashes
    from cgap_sbl_jadce.experiments.estimators import load_selection
    folder = tuning_path(cfg, method, '', pair)
    if pair in pending_tuning_resets(cfg, method):
        raise ValueError('An interrupted tuning reset must finish through Resume on its original request.')
    summary = json.loads((folder / 'summary.json').read_text())
    if summary.get('status') != 'complete' or not summary.get('outputs'):
        raise ValueError('Tuning is unfinished.')
    selected = load_selection(cfg, method, pair)
    files = [folder / name for name in summary['outputs']]
    if list(output_hashes(files).values()) != list(summary['outputs'].values()):
        raise ValueError('Saved tuning outputs are missing or changed; rerun this condition.')
    return selected


def reset_tuning_results(cfg, method, pairs, definitions, resume):
    from cgap_sbl_jadce.checkpoints import scientific_revision
    from cgap_sbl_jadce.compatibility import require_match
    from cgap_sbl_jadce.experiments.estimators import tuning_conditions, condition_config, ESTIMATORS
    allowed = set(tuning_conditions(cfg, pairs, method))
    if set(pairs) != allowed or len(pairs) != len(definitions):
        raise ValueError('Reset scope must match the requested tuning conditions.')
    folder = task_dir(cfg['run']['output_dir'], method) / 'tuning'
    if folder.is_symlink() or folder.parent.is_symlink():
        raise ValueError('Cannot reset results through a linked algorithm folder.')
    scope = hashlib.sha256(json.dumps(sorted(pairs)).encode()).hexdigest()[:16]
    marker = folder / '.resets' / f'{scope}.json'
    if marker.parent.is_symlink():
        raise ValueError('Cannot reset results through a linked reset folder.')
    recoveries = []
    job_id = os.environ.get('CGAP_JOB_DIRECTORY', '').rstrip('/').rsplit('/', 1)[-1]
    for path in (folder.parent / '.recovery').glob('tune*.json'):
        saved = json.loads(path.read_text())
        selected = saved['request'].get('conditions')
        scope_pairs = {tuple(pair) for pair in selected} if selected is not None else set(tuning_conditions(cfg, method=method))
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
        require_match(json.loads(marker.read_text()), identity, 'Pending tuning reset')
    targets = []
    for pair in pairs:
        target = tuning_path(cfg, method, '', pair)
        if target.is_symlink() or not (target == folder or target.resolve().parent == folder.resolve()):
            raise ValueError('Reset target is outside the selected algorithm run folder.')
        files = [target / name for name in ('summary.json', 'selected.json', 'thresholds.csv', 'trials.csv',
                                           'study.sqlite3', 'study.sqlite3-shm', 'study.sqlite3-wal',
                                           'sampler.pkl', 'search_progress.json')]

        directories = []
        for metadata in target.glob('.checkpoints/*/definition.json'):
            saved = json.loads(metadata.read_text())['definition']
            if (saved.get('method') != ESTIMATORS[method].name or
                    tuple(saved.get('condition', ())) != pair):
                raise ValueError('Checkpoint ownership differs from the selected condition.')
            directories.append(metadata.parent)
            files.extend([metadata, *metadata.parent.glob('*.npz')])
        if any(not path.resolve().is_relative_to(target.resolve()) for path in files):
            raise ValueError('A selected result file points outside its condition folder.')
        figure_root = tuning_figure_path(condition_config(cfg, pair, method), method, '')
        if figure_root.is_symlink() or not figure_root.resolve().is_relative_to(folder.parent.resolve()):
            raise ValueError('Tuning figures are outside the selected algorithm folder.')
        plots = [figure_root / f'{stem}.{suffix}' for stem in (
            f'{method}_threshold_scores', f'{method}_optuna_search_history', f'{method}_optuna_parameter_samples')
            for suffix in ('png', 'pdf')]
        if any(not path.resolve().is_relative_to(figure_root.resolve()) for path in plots):
            raise ValueError('A tuning figure points outside its condition folder.')
        targets.append((target, files + plots, directories))
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
    from cgap_sbl_jadce.figures.estimator_parameter_sweeps import plot_tuning_parameter_sweeps
    plot_tuning_parameter_sweeps(cfg, method)
    for path, saved, _, owned in recoveries:
        if owned:
            save_json(path, dict(saved, phase='calculating'))
    marker.unlink()
    if not any(marker.parent.iterdir()):
        marker.parent.rmdir()

