from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from cgap_sbl_jadce.paths import result_path, task_dir
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, save_npz as save_sparse_npz
from tqdm.auto import tqdm

from cgap_sbl_jadce.models.activity import generate_activity_dataset
from cgap_sbl_jadce.experiments.calibration import calibrate_kappa_adaptive, CalibrationBracketError
from cgap_sbl_jadce.config import resolve_activity_kappa, activity_spatial_dependence_parameters
from cgap_sbl_jadce.settings import calibration_settings, reference_settings, evaluation_settings
from cgap_sbl_jadce.models.spatial_dependence import build_C
from cgap_sbl_jadce.spatial_dependence_dataset import spatial_dependence_definition
from cgap_sbl_jadce.dataset import load_activity_samples, save_activity_samples
from cgap_sbl_jadce.io import ensure_dir, save_csv, save_json, atomic_path
from cgap_sbl_jadce.models.realizations import communication_realization
from cgap_sbl_jadce.communication_dataset import communication_definition, save_communication_sample
from cgap_sbl_jadce.reporting import print_task_report
from cgap_sbl_jadce.checkpoints import Checkpoint, checkpointed, is_resuming, output_hashes, completed_outputs
from cgap_sbl_jadce.provenance import save_task_snapshot


def output_dir(cfg: dict[str, Any]) -> Path:
    return ensure_dir(cfg["run"]["output_dir"])


@checkpointed
def calibrate_kappa_stage(cfg: dict[str, Any], show_progress: bool = True) -> dict[str, Any]:
    """kappa calibration: calibrate kappa, validate, and save effective run provenance."""
    cfg = deepcopy(cfg)
    sys = cfg["system"]
    act = {**cfg["activity"], **calibration_settings(cfg)}
    run = cfg["run"]
    D = float(act["D"])
    kappas = [float(value) for value in act.get("kappa_scan", [])]
    num_samples = int(act.get("calibration_samples", 2000))
    target = float(sys["S"])
    seed = int(cfg["seeds"]["kappa_calibration"])

    out = Path(run["output_dir"])
    caldir = ensure_dir(task_dir(out, 'kappa_calibration'))
    csv_path = caldir / "kappa_sweep.csv"
    summary_path = caldir / "summary.json"
    # Record the effective CLI values and defaults before executing calibration.
    cfg["system"]["S"] = target
    cfg["activity"].update({
        "D": D, "kappa_scan": kappas, "calibration_samples": num_samples,
        "kappa_refinement_steps": act.get("kappa_refinement_steps", [0.1, 0.02, 0.005]),
        "calibration_validation_samples": int(act.get("calibration_validation_samples", num_samples)),
        "calibration_tolerance": float(act.get("calibration_tolerance", 0.1)),
    })
    act = cfg["activity"]
    from cgap_sbl_jadce.compatibility import preparation_settings
    checkpoint = Checkpoint(caldir, {'task': 'kappa_calibration', 'settings': preparation_settings(cfg, 'kappa_calibration')})
    save_task_snapshot(cfg, 'kappa_calibration', status="started", calibration_seed=seed)
    try:
        _, best, records, details = calibrate_kappa_adaptive(
            checkpoint=checkpoint, target=target, coarse_values=kappas,
            refinement_steps=act["kappa_refinement_steps"],
            validation_seed=int(cfg["seeds"]["kappa_validation"]),
            validation_samples=act["calibration_validation_samples"],
            tolerance=act["calibration_tolerance"],
            seed=seed, num_samples=num_samples, N=int(sys["N"]), V=int(act["V"]),
            R=float(sys["R"]), D=D, batch_size=int(act.get("batch_size", 128)),
            show_progress=show_progress,
        )
    except CalibrationBracketError as error:
        save_csv(csv_path, error.records)
        save_json(summary_path, {"status": "failed", "error": str(error), "seed": seed})
        save_task_snapshot(cfg, 'kappa_calibration', status="failed", error=str(error), calibration_seed=seed)
        raise
    cfg["activity"]["kappa"] = best.kappa
    save_task_snapshot(cfg, 'kappa_calibration', status="success", calibration_seed=seed,
                       validation_seed=details["validation"]["seed"], selected_kappa=best.kappa)
    save_csv(csv_path, records)
    summary = {
        **details,
        "N": int(sys["N"]),
        "V": int(act["V"]),
        "R": float(sys["R"]),
        "D": D,
        "target_mean_active": target,
        "calibration_samples": num_samples,
        "seed": seed,
        "closest_grid_kappa": best.kappa,
        "closest_grid_mean_expected_active": best.mean_expected_active,
        "closest_grid_mean_realized_active": best.mean_realized_active,
        "absolute_expected_gap": abs(best.mean_expected_active - target),
    }
    from cgap_sbl_jadce.compatibility import preparation_settings
    summary["compatibility"] = preparation_settings(cfg, "kappa_calibration")
    save_json(summary_path, summary)

    print_task_report("KAPPA CALIBRATION — KAPPA CALIBRATION", {
        "D (m)": D,
        "target mean active S": target,
        "calibration realizations": num_samples,
        "closest grid kappa": best.kappa,
        "mean expected active": best.mean_expected_active,
        "mean realized active": best.mean_realized_active,
        "expected gap from target": best.mean_expected_active - target,
        "validation expected mean": details["validation"]["mean_expected_active"],
        "validation expected SE": details["validation"]["se_expected_active"],
        "validation within tolerance": details["validation"]["within_tolerance"],
    }, [csv_path, summary_path, out / "config.yaml", caldir / "manifest.json"])
    if not details["validation"]["within_tolerance"]:
        print("Validation mean exceeds the target tolerance; inspect the saved uncertainty before continuing.")
    return summary


def _activity_stats(samples: list[Any]) -> dict[str, float]:
    attempts = np.asarray([s.attempts for s in samples], dtype=float)
    active = np.asarray([np.sum(s.a) for s in samples], dtype=float)
    expected = np.asarray([np.sum(s.activation_probabilities) for s in samples], dtype=float)
    return {
        "samples": int(len(samples)),
        "mean_active": float(np.mean(active)),
        "std_active": float(np.std(active, ddof=1)) if len(active) > 1 else 0.0,
        "min_active": int(np.min(active)),
        "max_active": int(np.max(active)),
        "p05_active": float(np.percentile(active, 5)),
        "p50_active": float(np.percentile(active, 50)),
        "p95_active": float(np.percentile(active, 95)),
        "mean_expected_active": float(np.mean(expected)),
        "std_expected_active": float(np.std(expected, ddof=1)) if len(expected) > 1 else 0.0,
        "attempts_mean": float(np.mean(attempts)),
        "attempts_median": float(np.median(attempts)),
        "attempts_max": int(np.max(attempts)),
    }


@checkpointed
def generate_activity_stage(cfg: dict[str, Any], show_progress: bool = True) -> dict[str, Any]:
    kappa, kappa_source = resolve_activity_kappa(cfg)
    cfg = {**cfg, "activity": {**cfg["activity"], "kappa": kappa}}
    print(f"activity generation: using kappa={kappa:g} from {kappa_source}")
    cfg["task_runtime"] = {**cfg.get("task_runtime", {}), "kappa_source": kappa_source}
    out = output_dir(cfg)
    sys = cfg["system"]
    act = cfg["activity"]
    run = cfg["run"]
    common = dict(
        N=sys["N"], V=act["V"], R=sys["R"], kappa=act["kappa"], D=act["D"], S=sys["S"],
        exact_sparsity=act["exact_sparsity"], max_attempts=act.get("max_attempts", 100000),
        batch_size=act.get("batch_size", 128), show_progress=show_progress,
    )
    datasets = {
        split: generate_activity_dataset(
            seed=cfg["seeds"][f"activity_{split}"],
            num_samples=run[f"num_{split}_samples"],
            desc=f"activity generation {split} activity",
            checkpoint=Checkpoint(task_dir(out, 'activity') / split, {'task': 'activity', 'seed': cfg['seeds'][f'activity_{split}'],
                'samples': run[f'num_{split}_samples'], **{key: value for key, value in common.items() if key != 'show_progress'}}),
            **common,
        )
        for split in ("tuning", "evaluation")
    }
    tune, evaluation = datasets["tuning"], datasets["evaluation"]
    tune_path = result_path(out, "activity", "tuning_activity.npz")
    eval_path = result_path(out, "activity", "evaluation_activity.npz")
    for split, samples in datasets.items():
        save_activity_samples(result_path(out, "activity", f"{split}_activity.npz"), samples)
    realized_s_path = result_path(out, "activity", "realized_s.csv")
    realized_rows = []
    for split, samples in datasets.items():
        for sample_index, sample in enumerate(samples):
            realized_rows.append({
                "split": split,
                "sample": sample_index,
                "realized_S": int(np.sum(sample.a)),
                "expected_S": float(np.sum(sample.activation_probabilities)),
            })
    save_csv(realized_s_path, realized_rows)
    summary = {
        "N": int(sys["N"]), "target_mean_S": float(sys["S"]), "V": int(act["V"]), "R": float(sys["R"]),
        "kappa": float(act["kappa"]), "kappa_source": kappa_source, "D": float(act["D"]), "exact_sparsity": bool(act["exact_sparsity"]),
        "batch_size": int(act.get("batch_size", 128)),
        "tuning": _activity_stats(tune), "evaluation": _activity_stats(evaluation),
    }
    for split in datasets:
        summary[split]["mean_minus_target"] = summary[split]["mean_active"] - float(sys["S"])
    summary_path = result_path(out, "activity", "summary.json")
    from cgap_sbl_jadce.compatibility import preparation_settings
    summary["compatibility"] = {split: preparation_settings(cfg, "activity", split) for split in ("tuning", "evaluation")}
    save_json(summary_path, summary)
    save_task_snapshot(cfg, 'activity', show_progress=show_progress)
    print_task_report("ACTIVITY GENERATION — ACTIVITY MODEL", {
        "tuning samples": len(tune), "evaluation samples": len(evaluation),
        "target mean active S": float(sys["S"]),
        "mean active (tuning)": summary["tuning"]["mean_active"],
        "std active (tuning)": summary["tuning"]["std_active"],
        "range active (tuning)": f"{summary['tuning']['min_active']}..{summary['tuning']['max_active']}",
        "mean active (evaluation)": summary["evaluation"]["mean_active"],
        "std active (evaluation)": summary["evaluation"]["std_active"],
        "exact sparsity conditioning": bool(act["exact_sparsity"]),
    }, [tune_path, eval_path, realized_s_path, summary_path])
    return summary


@checkpointed
def communication_stage(cfg: dict[str, Any], show_progress: bool = True, conditions=None, methods=None) -> dict[str, Any]:
    """communication generation owns the full tuning and evaluation communication data."""
    if conditions is not None:
        from cgap_sbl_jadce.communication_dataset import add_communication_conditions
        return add_communication_conditions(cfg, conditions, methods or [], show_progress)
    from cgap_sbl_jadce.compatibility import validate_preparation
    for split in ("tuning", "evaluation"):
        validate_preparation(cfg, "activity", split)
    out = output_dir(cfg)
    summary = {"status": "started"}
    summary_path = result_path(out, "communication", "summary.json")
    save_json(summary_path, summary)
    save_task_snapshot(cfg, 'communication', status="started")
    for split in ("tuning", "evaluation"):
        definition = communication_definition(cfg, split)
        activities = load_activity_samples(result_path(out, "activity", f"{split}_activity.npz"))
        folder = task_dir(out, 'communication') / split
        checkpoint = Checkpoint(folder, definition)
        for q, activity in enumerate(tqdm(activities, desc=f"communication generation {split} communication",
                                        unit="realization", disable=not show_progress)):
            if len(activity.a) != definition["N"]:
                raise ValueError("Saved activity dimensions differ from configuration; rerun activity generation.")
            files = [folder / f'{q:04d}' / name for name in ('channels.npz', 'pilots.npz', 'observations.npz')]
            if completed_outputs(checkpoint, f'{q:04d}', files) is not None:
                continue
            realizations = {}
            for L, snr in definition["conditions"]:
                realizations[(L, snr)] = communication_realization(
                    activity, definition["seed"], q, definition["M"], L, snr)
            save_communication_sample(folder, q, realizations)
            checkpoint.save(f'{q:04d}', {'outputs': output_hashes(files)})
        summary[split] = {"samples": len(activities), "definition": definition,
                          "preview_condition": [int(reference_settings(cfg)["reference_pilot_length"] if split == "tuning" else cfg["evaluation"]["pilot_length"]), float(reference_settings(cfg)["reference_snr_db"] if split == "tuning" else evaluation_settings(cfg)["snr_db"])]}
    summary["status"] = "complete"
    save_json(summary_path, summary)
    save_task_snapshot(cfg, 'communication', status="complete", communication_datasets=summary)
    print_task_report("COMMUNICATION GENERATION — COMMUNICATION DATASETS", {
        "tuning realizations": summary["tuning"]["samples"],
        "evaluation realizations": summary["evaluation"]["samples"],
        "unique evaluation conditions": len(summary["evaluation"]["definition"]["conditions"]),
    }, [task_dir(out, 'communication') / "tuning", task_dir(out, 'communication') / "evaluation", summary_path])
    return summary



@checkpointed
def spatial_dependence_stage(cfg: dict[str, Any], show_progress: bool = True) -> dict[str, Any]:
    """spatial dependence saves one CSR dependence matrix per tuning/evaluation geometry."""
    from cgap_sbl_jadce.compatibility import validate_preparation
    for split in ("tuning", "evaluation"):
        validate_preparation(cfg, "activity", split)
    out = output_dir(cfg)
    summary = {"status": "started"}
    summary_path = result_path(out, "spatial_dependence", "summary.json")
    save_json(summary_path, summary)
    save_task_snapshot(cfg, 'spatial_dependence', status="started")
    for split in ("tuning", "evaluation"):
        definition = spatial_dependence_definition(cfg, split)
        activities = load_activity_samples(result_path(out, "activity", f"{split}_activity.npz"))
        folder = ensure_dir(task_dir(out, 'spatial_dependence') / split)
        checkpoint = Checkpoint(folder, definition)
        nonzeros = []
        for q, activity in enumerate(tqdm(activities, desc=f"spatial dependence {split} spatial matrices",
                                        unit="realization", disable=not show_progress)):
            if len(activity.a) != definition["N"]:
                raise ValueError("Saved activity dimensions differ from configuration; rerun activity generation.")
            path = folder / f"{q:04d}.npz"
            cached = completed_outputs(checkpoint, f'{q:04d}', [path])
            if cached is not None:
                nonzeros.append(cached['nonzeros'])
                continue
            C = csr_matrix(build_C(activity.device_locations, rho=definition["rho"], U=definition["U"]))
            with atomic_path(path) as temporary:
                save_sparse_npz(temporary, C, compressed=True)
            checkpoint.save(f'{q:04d}', {'nonzeros': C.nnz, 'outputs': output_hashes([path])})
            nonzeros.append(C.nnz)
        summary[split] = {"samples": len(activities), "definition": definition,
                          "mean_nonzeros": float(np.mean(nonzeros)),
                          "mean_density": float(np.mean(nonzeros) / definition["N"]**2)}
    summary["status"] = "complete"
    save_json(summary_path, summary)
    save_task_snapshot(cfg, 'spatial_dependence', status="complete", spatial_dependence_datasets=summary)
    print_task_report("SPATIAL DEPENDENCE — SPATIAL DEPENDENCE MATRICES", {
        "tuning matrices": summary["tuning"]["samples"],
        "shared evaluation matrices": summary["evaluation"]["samples"],
        "rho (m)": definition["rho"], "U (m)": definition["U"], "format": "compressed sparse CSR",
    }, [task_dir(out, 'spatial_dependence') / "tuning", task_dir(out, 'spatial_dependence') / "evaluation", summary_path])
    return summary
