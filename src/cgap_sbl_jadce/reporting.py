from __future__ import annotations

import json
from pathlib import Path
from cgap_sbl_jadce.paths import result_path, figure_path
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


def report_task(output_dir: str | Path, task: int) -> None:
    """Print saved analysis artifacts for a completed scientific task (01-09)."""
    out = Path(output_dir)
    files = {
        1: result_path(out, "activity_calibration", "summary.json"),
        2: result_path(out, "activity", "summary.json"),
        3: result_path(out, "communication", "summary.json"),
        4: result_path(out, "correlation", "summary.json"),
        5: result_path(out, "tuning", "alpha_beta_summary.json"),
        6: result_path(out, "tuning", "selected.json"),
        7: result_path(out, "evaluation", "run_summary.json"),
        8: result_path(out, "convergence", "summary.json"),
        9: result_path(out, "evaluation", "summary.json"),
    }
    if task not in files:
        raise ValueError("task must be an integer from 1 through 9")
    path = files[task]
    if not path.exists():
        raise FileNotFoundError(f"Task {task:02d} analysis is not available yet: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Task {task:02d} saved analysis: {path}")
    print(json.dumps(data, indent=2, sort_keys=False))
