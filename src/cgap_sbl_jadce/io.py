from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path); ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f: json.dump(data, f, indent=2, sort_keys=True)


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path); ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f: yaml.safe_dump(data, f, sort_keys=False)


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path); ensure_dir(p.parent)
    pd.DataFrame(rows).to_csv(p, index=False)


def save_npz(path: str | Path, **arrays: Any) -> None:
    p = Path(path); ensure_dir(p.parent)
    np.savez_compressed(p, **arrays)
