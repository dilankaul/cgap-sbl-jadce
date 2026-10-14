from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
from functools import lru_cache
import json
import os
import platform
from zipfile import BadZipFile

import numpy as np
import scipy

from cgap_sbl_jadce.io import save_json, save_npz
from cgap_sbl_jadce.execution_control import pause_point


_session = ContextVar('checkpoint_session', default=None)


def pending_checkpoint_files():
    session = _session.get()
    return [(checkpoint.folder / name, checkpoint.digest)
            for checkpoint in (session['checkpoints'] if session else [])
            for name in list(checkpoint.units) + ['definition.json']
            if (checkpoint.folder / name).exists()]


def is_resuming():
    return bool(_session.get() and _session.get()['resume'])


@contextmanager
def checkpoint_session(resume=False):
    session = {'resume': resume, 'checkpoints': []}
    token = _session.set(session)
    try:
        yield
    except BaseException:
        raise
    else:
        for checkpoint in reversed(session['checkpoints']):
            checkpoint.clear(prune=True)
    finally:
        _session.reset(token)


def checkpointed(stage):
    @wraps(stage)
    def run(*args, resume=None, **kwargs):
        if _session.get() is not None and resume is None:
            return stage(*args, **kwargs)
        with checkpoint_session(resume=bool(resume)):
            return stage(*args, **kwargs)
    return run


def scientific_revision():
    root = Path(__file__).parent
    files = list((root / 'models').glob('*.py')) + list((root / 'algorithms').glob('*.py'))
    files += list((root / 'experiments').glob('*.py')) + [root / name for name in ('metrics.py', 'checkpoints.py', 'additional_conditions.py', 'communication_dataset.py', 'spatial_dependence_dataset.py', 'condition_results.py', 'settings.py', 'config.py', 'compatibility.py')]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_owner(definition):
    if 'condition' in definition and 'definition' in definition:
        return ('run', definition['definition'].get('method'), tuple(definition['condition']))
    if 'method' in definition and 'condition' in definition:
        return ('tune', definition['method'], tuple(definition['condition']))
    if 'task' in definition:
        return ('task', definition['task'])
    if 'q' in definition and 'methods' in definition:
        return ('convergence', definition['q'], definition['L'], definition['snr'], tuple(sorted(definition['methods'])))
    return None


def prune_owned_checkpoints(root, definition):
    owner = checkpoint_owner(definition)
    if owner is None:
        return
    root = Path(root)
    for manifest in list(root.glob('*/definition.json')):
        try:
            identity = json.loads(manifest.read_text())
            if checkpoint_owner(identity['definition']) != owner:
                continue

            children = list(manifest.parent.rglob('definition.json'))
            for child in sorted(children, key=lambda path: len(path.parts), reverse=True):
                digest = child.parent.name
                for path in child.parent.glob('*.npz'):
                    try:
                        with np.load(path, allow_pickle=False) as archive:
                            matches = str(archive['_definition']) == digest
                    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                        matches = False
                    if matches:
                        path.unlink()
                if not any(child.parent.glob('*.npz')):
                    child.unlink(missing_ok=True)

            for folder in sorted(manifest.parent.rglob('*'), key=lambda path: len(path.parts), reverse=True):
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
            if manifest.parent.exists() and not any(manifest.parent.iterdir()):
                manifest.parent.rmdir()
        except (OSError, ValueError, KeyError):
            continue
    if root.exists() and not any(root.iterdir()):
        root.rmdir()


class Checkpoint:
    def __init__(self, folder, definition):
        identity = {'schema': 1, 'definition': definition, 'code': scientific_revision(),
                    'numpy': np.__version__, 'scipy': scipy.__version__, 'machine': platform.machine(),
                    'threads': {key: os.environ.get(key) for key in (
                        'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
        self.definition = definition
        self.digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.folder = Path(folder) / '.checkpoints' / self.digest
        self.units = set()
        session = _session.get()
        if session is not None:
            if not session['resume']:


                for path in self.folder.glob('*.npz'):
                    try:
                        with np.load(path, allow_pickle=False) as archive:
                            if str(archive['_definition']) == self.digest:
                                self.units.add(path.name)
                    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                        pass
                self.clear()
            elif not self.folder.exists() and any(self.folder.parent.glob('*/**/*.npz')):
                raise ValueError('Checkpoint settings, inputs, code or numerical environment changed. Choose a fresh rerun.')
            session['checkpoints'].append(self)
        self.folder.mkdir(parents=True, exist_ok=True)
        save_json(self.folder / 'definition.json', identity)

    def clear(self, prune=False):
        for name in self.units:
            (self.folder / name).unlink(missing_ok=True)
        self.units.clear()
        (self.folder / 'definition.json').unlink(missing_ok=True)
        if prune:
            prune_owned_checkpoints(self.folder.parent, self.definition)
        for folder in (self.folder, self.folder.parent):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()

    def child(self, name, definition):
        return Checkpoint(self.folder / name, definition)

    def load(self, key):
        pause_point()
        path = self.folder / f'{key}.npz'
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                if str(archive['_definition']) != self.digest:
                    raise ValueError('Checkpoint identity differs')
                metadata = json.loads(str(archive['_metadata']))
                arrays = {name: archive[name] for name in archive.files if not name.startswith('_')}
            self.units.add(path.name)
            return metadata, arrays
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            print(f'Ignoring incomplete checkpoint unit: {path.name}', flush=True)
            return None

    def save(self, key, metadata=None, **arrays):
        save_npz(self.folder / f'{key}.npz', _definition=np.asarray(self.digest),
                 _metadata=np.asarray(json.dumps(metadata)), **arrays)
        self.units.add(f'{key}.npz')
        pause_point()


def output_hashes(paths):
    hashes = {}
    for path in paths:
        path = Path(path)
        with path.open('rb') as stream:
            hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return hashes


def completed_outputs(checkpoint, key, paths):
    cached = checkpoint.load(key)
    if cached is None:
        return None
    metadata, arrays = cached
    if not isinstance(metadata, dict):
        return None
    try:
        return metadata if metadata.get('outputs') == output_hashes(paths) else None
    except OSError:
        return None


@lru_cache(maxsize=16384)
def _archive_digest(path, size, mtime, ctime):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').digest()


def archive_identity(folder, count, names):
    folder = Path(folder)
    digest = hashlib.sha256()
    for q in range(count):
        for name in names:
            relative = f'{q:04d}/{name}' if name != 'matrix' else f'{q:04d}.npz'
            digest.update(relative.encode())
            path = folder / relative
            info = path.stat()
            digest.update(_archive_digest(str(path.resolve()), info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return digest.hexdigest()
