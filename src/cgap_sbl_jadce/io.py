from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import numpy as np
import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def atomic_path(path: str | Path):
    target = Path(path)
    ensure_dir(target.parent)
    fd, name = tempfile.mkstemp(prefix='.tmp-', suffix=target.suffix, dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    with atomic_path(path) as p:
        p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with atomic_path(path) as p:
        p.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def save_csv(path: str | Path, rows) -> None:
    with atomic_path(path) as p:
        pd.DataFrame(rows).to_csv(p, index=False)


def save_npz(path: str | Path, **arrays: Any) -> None:
    with atomic_path(path) as p:
        np.savez_compressed(p, **arrays)
