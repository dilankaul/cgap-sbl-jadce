from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
from pathlib import Path


_snapshot = ContextVar('validation_snapshot', default=None)


@contextmanager
def validation_snapshot():
    if _snapshot.get() is not None:
        yield
        return
    token = _snapshot.set({})
    try:
        yield
    finally:
        _snapshot.reset(token)


def snapshot_cached(operation):
    @wraps(operation)
    def read(*args, **kwargs):
        snapshot = _snapshot.get()
        if snapshot is None:
            return operation(*args, **kwargs)
        key = (operation, json.dumps([args, kwargs], sort_keys=True, default=str))
        if key not in snapshot:
            snapshot[key] = operation(*args, **kwargs)
        return snapshot[key]
    return read


def file_version(path):
    path = str(path)
    snapshot = _snapshot.get()
    key = ('file', path)
    if snapshot is not None and key in snapshot:
        return snapshot[key]
    stat = Path(path).stat()
    version = path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    if snapshot is not None:
        snapshot[key] = version
    return version
