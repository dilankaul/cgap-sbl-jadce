from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def print_task_report(task: str, summary: dict[str, Any], files: Iterable[str | Path] = ()) -> None:
    print(f"\n{'=' * 72}\n{task} COMPLETE\n{'=' * 72}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:28s}: {value:.6g}")
        else:
            print(f"{key:28s}: {value}")
    file_list = [str(Path(f)) for f in files]
    if file_list:
        print("Saved files:")
        for f in file_list:
            print(f"  - {f}")
    print()
