"""Persistent, definition-specific checkpoints for completed calculation units."""
from pathlib import Path
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


def scientific_revision():
    root = Path(__file__).parent
    files = list((root / 'models').glob('*.py')) + list((root / 'algorithms').glob('*.py'))
    files += list((root / 'experiments').glob('*.py')) + [root / 'metrics.py', root / 'checkpoints.py']
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Checkpoint:
    """A unit is reusable only after its whole archive has been committed."""
    def __init__(self, folder, definition):
        identity = {'schema': 1, 'definition': definition, 'code': scientific_revision(),
                    'numpy': np.__version__, 'scipy': scipy.__version__, 'machine': platform.machine(),
                    'threads': {key: os.environ.get(key) for key in (
                        'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
        self.digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.folder = Path(folder) / '.checkpoints' / self.digest
        self.folder.mkdir(parents=True, exist_ok=True)
        save_json(self.folder / 'definition.json', identity)

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
            return metadata, arrays
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            print(f'Ignoring incomplete checkpoint unit: {path.name}', flush=True)
            return None

    def save(self, key, metadata=None, **arrays):
        save_npz(self.folder / f'{key}.npz', _definition=np.asarray(self.digest),
                 _metadata=np.asarray(json.dumps(metadata)), **arrays)
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
    """Reuse a byte digest only while the file identity remains unchanged."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').digest()


def archive_identity(folder, count, names):
    """Identify the actual archive bytes, not only their generating settings."""
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
