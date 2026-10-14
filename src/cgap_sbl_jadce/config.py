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
        )
        seeds = cfg["seeds"]
        if not isinstance(seeds, dict) or set(seeds) != set(names):
            raise ValueError("seeds must define exactly: " + ", ".join(names))
        values = [seeds[name] for name in names]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Each seed must be a non-negative integer.")
        if len(set(values)) != len(values):
            raise ValueError("Each experimental random stream must have a distinct seed.")
    if cfg.get('experiment', {}).get('name'):

        cfg['run']['output_dir'] = str(Path(path).resolve().parent)
    return cfg


def resolve_activity_kappa(cfg: dict[str, Any]) -> tuple[float, str]:
    import json
    import math
    from cgap_sbl_jadce.paths import result_path

    activity = cfg["activity"]
    source = activity.get("kappa_source", "configured")
    if source == "configured":
        kappa = float(activity["kappa"])
        origin = "configured"
    elif source == "calibration":
        path = result_path(cfg["run"]["output_dir"], "kappa_calibration", "summary.json")
        if not path.exists():
            raise FileNotFoundError(f"Run kappa calibration (calibrate kappa) in the GUI first: {path}")
        with path.open(encoding="utf-8") as f:
            summary = json.load(f)
        if summary.get("status", "success") != "success":
            raise ValueError("kappa calibration calibration failed; rerun kappa calibration in the GUI")
        expected = {
            "N": cfg["system"]["N"], "R": cfg["system"]["R"],
            "V": activity["V"], "D": activity["D"],
            "target_mean_active": cfg["system"]["S"],
            "seed": int(cfg["seeds"]["kappa_calibration"]),
        }
        mismatches = [key for key, value in expected.items() if summary.get(key) != value]
        if mismatches:
            raise ValueError(f"kappa calibration calibration differs in {', '.join(mismatches)}; rerun kappa calibration in the GUI for this setup.")
        from cgap_sbl_jadce.compatibility import validate_preparation
        validate_preparation(cfg, "kappa_calibration")
        kappa = float(summary["closest_grid_kappa"])
        origin = str(path)
    else:
        raise ValueError("activity.kappa_source must be configured or calibration")
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("Selected kappa must be finite and positive")
    return kappa, origin


def activity_spatial_dependence_parameters(cfg: dict[str, Any]) -> dict[str, float]:
    import json
    from cgap_sbl_jadce.paths import result_path

    path = result_path(cfg["run"]["output_dir"], "activity", "summary.json")
    if path.exists():
        summary = json.loads(path.read_text())
        return {"rho": float(summary["kappa"]), "U": float(summary["D"])}
    return {"rho": resolve_activity_kappa(cfg)[0], "U": float(cfg["activity"]["D"])}
