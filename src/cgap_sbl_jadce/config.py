from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping.")
    if "seeds" in cfg:
        names = (
            "kappa_calibration", "kappa_validation", "activity_tuning",
            "activity_evaluation", "communication_tuning", "communication_evaluation",
            "spatial_randomization",
        )
        seeds = cfg["seeds"]
        if not isinstance(seeds, dict) or set(seeds) != set(names):
            raise ValueError("seeds must define exactly: " + ", ".join(names))
        values = [seeds[name] for name in names]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Each seed must be a non-negative integer.")
        if len(set(values)) != len(values):
            raise ValueError("Each experimental random stream must have a distinct seed.")
    return cfg


def resolve_activity_kappa(cfg: dict[str, Any]) -> tuple[float, str]:
    """Read a compatible Task-01 selection without rewriting the input config."""
    import json
    import math
    from cgap_sbl_jadce.paths import result_path

    activity = cfg["activity"]
    source = activity.get("kappa_source", "configured")
    if source == "configured":
        kappa = float(activity["kappa"])
        origin = "configured"
    elif source == "calibration":
        path = result_path(cfg["run"]["output_dir"], "activity_calibration", "summary.json")
        if not path.exists():
            raise FileNotFoundError(f"Run Task 01 (01_calibrate_kappa.py) first: {path}")
        with path.open(encoding="utf-8") as f:
            summary = json.load(f)
        if summary.get("status", "success") != "success":
            raise ValueError("Task 01 calibration failed; rerun 01_calibrate_kappa.py")
        expected = {
            "N": cfg["system"]["N"], "R": cfg["system"]["R"],
            "V": activity["V"], "D": activity["D"],
            "target_mean_active": cfg["system"]["S"],
            "seed": int(cfg["seeds"]["kappa_calibration"]),
        }
        mismatches = [key for key, value in expected.items() if summary.get(key) != value]
        if mismatches:
            raise ValueError(f"Task 01 calibration differs in {', '.join(mismatches)}; rerun 01_calibrate_kappa.py for this configuration.")
        kappa = float(summary["closest_grid_kappa"])
        origin = str(path)
    else:
        raise ValueError("activity.kappa_source must be configured or calibration")
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("Selected kappa must be finite and positive")
    return kappa, origin


def activity_correlation_parameters(cfg: dict[str, Any]) -> dict[str, float]:
    """Paper VIII-A: rho=kappa and U=D, using the saved activity model."""
    import json
    from cgap_sbl_jadce.paths import result_path

    path = result_path(cfg["run"]["output_dir"], "activity", "summary.json")
    if path.exists():
        summary = json.loads(path.read_text())
        return {"rho": float(summary["kappa"]), "U": float(summary["D"])}
    return {"rho": resolve_activity_kappa(cfg)[0], "U": float(cfg["activity"]["D"])}
