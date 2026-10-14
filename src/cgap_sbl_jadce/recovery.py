"""Portable action requests; local PIDs and logs remain in .gui/."""
import hashlib
import json
import os
import time
from pathlib import Path

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.io import save_json
from cgap_sbl_jadce.paths import task_dir


def request_scope(request):
    selection = {key: request[key] for key in ('conditions', 'methods', 'condition', 'realization_index') if key in request}
    if not selection:
        return ''
    for key in ('conditions', 'methods'):
        if key in selection:
            selection[key] = sorted(selection[key])
    return '_' + hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()[:16]


def recovery_path(config, task, action='run', request=None):
    folder = task_dir(load_config(config)['run']['output_dir'], task) / '.recovery'
    if request is None:
        paths = sorted(folder.glob(f'{action}*.json'), key=lambda path: path.stat().st_mtime_ns, reverse=True)
        if paths:
            return paths[0]
    return folder / f'{action}{request_scope(request or {})}.json'


def load_recoveries(config, task, action='run'):
    folder = task_dir(load_config(config)['run']['output_dir'], task) / '.recovery'
    result = []
    for path in folder.glob(f'{action}*.json'):
        saved = json.loads(path.read_text())
        request = dict(saved['request'], config=str(Path(config).resolve()))
        result.append({**saved, 'request': request, 'path': path})
    return sorted(result, key=lambda saved: saved['created'], reverse=True)


def load_recovery(config, task, action='run', request=None):
    records = load_recoveries(config, task, action)
    if request is not None:
        records = [record for record in records if request_scope(record['request']) == request_scope(request)]
    return records[0] if records else None


def validate_recovery(request):
    from cgap_sbl_jadce.gui.inputs import input_signature
    saved = load_recovery(request['config'], request['task'], request.get('action', 'run'), request)
    if saved is None:
        raise ValueError('No unfinished recovery data remain. Choose Start or Rerun for a fresh calculation.')
    from cgap_sbl_jadce.gui.jobs import config_digest
    if config_digest(request['config'], request if saved.get('schema') == 2 else None) != saved['config_digest']:
        raise ValueError('Configuration changed. Choose a fresh rerun.')
    if input_signature(request, portable=True) != saved['input_signature']:
        raise ValueError('Checkpoint inputs or scientific code changed. Choose a fresh rerun.')
    if {key: value for key, value in request.items() if key != 'config'} != {
            key: value for key, value in saved['request'].items() if key != 'config'}:
        raise ValueError('Resume must use the saved task and condition selection.')
    return saved


def begin_recovery(request, resume=False):
    from cgap_sbl_jadce.gui.inputs import input_signature
    path = recovery_path(request['config'], request['task'], request.get('action', 'run'), request)
    if resume:
        validate_recovery(request)
        return
    portable = dict(request)
    portable['config'] = os.path.relpath(Path(request['config']).resolve(), path.parent)
    from cgap_sbl_jadce.gui.jobs import config_digest
    save_json(path, {'schema': 2, 'request': portable, 'created': time.time(),
                     'job_id': Path(os.environ.get('CGAP_JOB_DIRECTORY', '')).name,
                     'config_digest': config_digest(request['config'], request),
                     'input_signature': input_signature(request, portable=True)})


def finish_recovery(request):
    path = recovery_path(request['config'], request['task'], request.get('action', 'run'), request)
    path.unlink(missing_ok=True)
    if path.parent.is_dir() and not any(path.parent.iterdir()):
        path.parent.rmdir()


def record_completed_outputs(request, figure_paths):
    """Commit final outputs before releasing checkpoints, closing the crash window."""
    from cgap_sbl_jadce.checkpoints import pending_checkpoint_files
    saved = load_recovery(request['config'], request['task'], request.get('action', 'run'), request)
    if saved is None:
        return
    cfg = load_config(request['config'])
    root = Path(cfg['run']['output_dir']).resolve()
    task = task_dir(root, request['task'])
    if request.get('action') == 'tune':
        from cgap_sbl_jadce.experiments.estimators import tuning_conditions, condition_config
        from cgap_sbl_jadce.paths import tuning_path, tuning_figure_path
        pairs = tuning_conditions(cfg, request.get('conditions'), request['task'])
        sources = [source for pair in pairs for source in (
            tuning_path(condition_config(cfg, pair, request['task']), request['task'], ''),
            tuning_figure_path(condition_config(cfg, pair, request['task']), request['task'], ''))]
    elif request['task'] in {'omp', 'cosamp', 'sbl', 'anc'}:
        from cgap_sbl_jadce.experiments.estimators import evaluation_conditions
        from cgap_sbl_jadce.paths import run_path
        pairs = list(dict.fromkeys(value[:2] for value in evaluation_conditions(cfg, request.get('conditions'), method=request['task'])))
        sources = [run_path(cfg, request['task'], condition=pair) for pair in pairs]
    else:
        sources = [task]
    files = set(Path(path).resolve() for path in figure_paths
                if any(Path(path).resolve().is_relative_to(source.resolve()) for source in sources))
    for source in sources:
        files.update(path.resolve() for path in source.rglob('*') if path.is_file() and
                     '.checkpoints' not in path.parts and '.recovery' not in path.parts and
                     not path.name.startswith('.tmp-'))
    outputs = {}
    for path in files:
        with path.open('rb') as stream:
            outputs[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    portable = {key: value for key, value in saved.items() if key not in {'path', 'request'}}
    portable['request'] = dict(saved['request'], config=os.path.relpath(Path(request['config']).resolve(), saved['path'].parent))
    portable.update(outputs=outputs, phase='results_saved',
                    figures=[Path(path).resolve().relative_to(root).as_posix() for path in figure_paths],
                    checkpoint_files=[{'path': path.resolve().relative_to(root).as_posix(), 'identity': identity}
                                      for path, identity in pending_checkpoint_files()])
    save_json(saved['path'], portable)


def restore_completed_outputs(request):
    """Finish an interrupted cleanup without repeating a completed task."""
    import numpy as np
    saved = validate_recovery(request)
    if saved.get('phase') != 'results_saved':
        return None
    root = Path(load_config(request['config'])['run']['output_dir']).resolve()
    for name, expected in saved['outputs'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Recovery output is outside this setup.')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('Completed outputs changed during interruption. Choose a fresh rerun.')
    for unit in saved['checkpoint_files']:
        path = (root / unit['path']).resolve()
        if not path.is_relative_to(root) or '.checkpoints' not in path.parts:
            raise ValueError('Checkpoint recovery path is outside its setup.')
        if not path.exists():
            continue
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as archive:
                if str(archive['_definition']) != unit['identity']:
                    raise ValueError('Checkpoint identity changed during cleanup.')
        elif path.name != 'definition.json' or path.parent.name != unit['identity']:
            raise ValueError('Unexpected checkpoint metadata path.')
        path.unlink()
        for folder in (path.parent, path.parent.parent):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
    return [root / path for path in saved['figures']]
