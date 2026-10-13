from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from copy import deepcopy
from cgap_sbl_jadce.paths import result_path, task_dir
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz as save_sparse_npz
from tqdm.auto import tqdm

from cgap_sbl_jadce.models.activity import generate_activity_dataset, randomize_activity_sample
from cgap_sbl_jadce.experiments.calibration import calibrate_kappa_adaptive, CalibrationBracketError
from cgap_sbl_jadce.algorithms.cgap_sbl import cgap_sbl_anc
from cgap_sbl_jadce.algorithms.cosamp import mmv_cosamp
from cgap_sbl_jadce.algorithms.omp import mmv_omp
from cgap_sbl_jadce.algorithms.sbl import sbl
from cgap_sbl_jadce.config import resolve_activity_kappa, activity_spatial_dependence_parameters
from cgap_sbl_jadce.models.spatial_dependence import build_C
from cgap_sbl_jadce.spatial_dependence_dataset import (spatial_dependence_definition, validate_spatial_dependence_dataset,
    load_spatial_dependence_matrix, load_tuning_spatial_dependence)
from cgap_sbl_jadce.dataset import load_activity_samples, save_activity_samples
from cgap_sbl_jadce.io import ensure_dir, save_csv, save_json, save_npz
from cgap_sbl_jadce.metrics import NMSE_SCOPE, nmse, oracle_sparsity_budget, precision_recall_f1, support_from_gamma, support_from_rows
from cgap_sbl_jadce.models.realizations import communication_realization, paired_activity_realizations
from cgap_sbl_jadce.communication_dataset import (communication_definition, validate_communication_dataset,
    save_communication_sample, load_communication_sample, load_tuning_communication)
from cgap_sbl_jadce.reporting import print_task_report
from cgap_sbl_jadce.provenance import save_task_snapshot
from cgap_sbl_jadce.experiments.tuning import threshold_curve


def output_dir(cfg: dict[str, Any]) -> Path:
    return ensure_dir(cfg["run"]["output_dir"])


def tuning_definition(cfg: dict[str, Any]) -> dict[str, Any]:
    """Identify the mathematics and experiment used to select parameters."""
    return {"method": "CGAP-SBL-ANC", "revision": 3, "nmse_scope": NMSE_SCOPE, "tau_grid": {"start": 0.0, "stop": 0.2, "step": 0.001},
            "algorithms": cfg["algorithms"], "spatial_dependence": activity_spatial_dependence_parameters(cfg),
            "tuning": cfg["tuning"], "system": cfg["system"],
            "tuning_samples": cfg["run"]["num_tuning_samples"],
            "activity_seed": cfg["seeds"]["activity_tuning"],
            "communication_seed": cfg["seeds"]["communication_tuning"]}


def validate_selection(selected: dict, cfg: dict) -> None:
    if selected.get("definition") != tuning_definition(cfg):
        raise ValueError("Saved tuning belongs to a different method or setup. Rerun Task 05 for CGAP-SBL-ANC.")


def calibrate_kappa_stage(cfg: dict[str, Any], show_progress: bool = True) -> dict[str, Any]:
    """Task 01: calibrate kappa, validate, and save effective run provenance."""
    cfg = deepcopy(cfg)
    sys = cfg["system"]
    act = cfg["activity"]
    run = cfg["run"]
    D = float(act["D"])
    kappas = [float(value) for value in act.get("kappa_scan", [])]
    num_samples = int(act.get("calibration_samples", 2000))
    target = float(sys["S"])
    seed = int(cfg["seeds"]["kappa_calibration"])

    out = Path(run["output_dir"])
    caldir = ensure_dir(task_dir(out, 1))
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
    save_task_snapshot(cfg, 1, status="started", calibration_seed=seed)
    try:
        _, best, records, details = calibrate_kappa_adaptive(
            target=target, coarse_values=kappas,
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
        save_task_snapshot(cfg, 1, status="failed", error=str(error), calibration_seed=seed)
        raise
    cfg["activity"]["kappa"] = best.kappa
    save_task_snapshot(cfg, 1, status="success", calibration_seed=seed,
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
    save_json(summary_path, summary)

    print_task_report("TASK 01 — KAPPA CALIBRATION", {
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


def generate_activity_stage(cfg: dict[str, Any], show_progress: bool = True) -> dict[str, Any]:
    kappa, kappa_source = resolve_activity_kappa(cfg)
    cfg = {**cfg, "activity": {**cfg["activity"], "kappa": kappa}}
    print(f"Task 02: using kappa={kappa:g} from {kappa_source}")
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
            desc=f"Task 02 {split} activity", **common,
        )
        for split in ("tuning", "evaluation")
    }
    datasets["randomized_evaluation"] = [
        randomize_activity_sample(sample, cfg["seeds"]["spatial_randomization"], q)
        for q, sample in enumerate(tqdm(datasets["evaluation"], desc="Task 02 randomized activity",
                                        unit="realization", disable=not show_progress))
    ]
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
        "randomized_evaluation": {**_activity_stats(datasets["randomized_evaluation"]),
            "seed": cfg["seeds"]["spatial_randomization"],
            "rule": "Uniform S_q distinct devices without replacement, paired by realization index"},
    }
    for split in datasets:
        summary[split]["mean_minus_target"] = summary[split]["mean_active"] - float(sys["S"])
    summary_path = result_path(out, "activity", "summary.json")
    save_json(summary_path, summary)
    save_task_snapshot(cfg, 2, show_progress=show_progress)
    print_task_report("TASK 02 — ACTIVITY MODEL", {
        "tuning samples": len(tune), "evaluation samples": len(evaluation),
        "target mean active S": float(sys["S"]),
        "mean active (tuning)": summary["tuning"]["mean_active"],
        "std active (tuning)": summary["tuning"]["std_active"],
        "range active (tuning)": f"{summary['tuning']['min_active']}..{summary['tuning']['max_active']}",
        "mean active (evaluation)": summary["evaluation"]["mean_active"],
        "std active (evaluation)": summary["evaluation"]["std_active"],
        "exact sparsity conditioning": bool(act["exact_sparsity"]),
    }, [tune_path, eval_path, result_path(out, "activity", "randomized_evaluation_activity.npz"), realized_s_path, summary_path])
    return summary


def communication_stage(cfg: dict[str, Any], show_progress: bool = True) -> dict[str, Any]:
    """Task 03 owns the full tuning and paired evaluation communication data."""
    out = output_dir(cfg)
    summary = {"status": "started"}
    summary_path = result_path(out, "communication", "summary.json")
    save_json(summary_path, summary)
    save_task_snapshot(cfg, 3, status="started")
    for split in ("tuning", "evaluation"):
        definition = communication_definition(cfg, split)
        activities = load_activity_samples(result_path(out, "activity", f"{split}_activity.npz"))
        randomized = None
        if split == "evaluation":
            randomized = load_activity_samples(result_path(out, "activity", "randomized_evaluation_activity.npz"))
            if len(randomized) != len(activities):
                raise ValueError("Paired activity lengths differ; rerun Task 02.")
        folder = task_dir(out, 3) / split
        for q, activity in enumerate(tqdm(activities, desc=f"Task 03 {split} communication",
                                        unit="realization", disable=not show_progress)):
            if len(activity.a) != definition["N"]:
                raise ValueError("Saved activity dimensions differ from configuration; rerun Task 02.")
            realizations = {}
            for L, snr in definition["conditions"]:
                paired = ({"correlated": communication_realization(activity, definition["seed"], q, definition["M"], L, snr)}
                          if randomized is None else paired_activity_realizations(
                              activity, randomized[q], definition["seed"], q, definition["M"], L, snr))
                realizations.update({(scenario, L, snr): r for scenario, r in paired.items()})
            save_communication_sample(folder, q, realizations)
        summary[split] = {"samples": len(activities), "definition": definition,
                          "preview_condition": [int(cfg["tuning"]["reference_pilot_length"] if split == "tuning" else cfg["evaluation"]["pilot_length"]), float(cfg["tuning"]["reference_snr_db"])],
                          "scenarios": ["correlated"] if randomized is None else ["correlated", "randomized"]}
    summary["status"] = "complete"
    save_json(summary_path, summary)
    save_task_snapshot(cfg, 3, status="complete", communication_datasets=summary)
    print_task_report("TASK 03 — COMMUNICATION DATASETS", {
        "tuning realizations": summary["tuning"]["samples"],
        "evaluation pairs": summary["evaluation"]["samples"],
        "unique evaluation conditions": len(summary["evaluation"]["definition"]["conditions"]),
    }, [task_dir(out, 3) / "tuning", task_dir(out, 3) / "evaluation", summary_path])
    return summary



def spatial_dependence_stage(cfg: dict[str, Any], show_progress: bool = True) -> dict[str, Any]:
    """Task 04 saves one CSR dependence matrix per tuning/evaluation geometry."""
    out = output_dir(cfg)
    summary = {"status": "started"}
    summary_path = result_path(out, "spatial_dependence", "summary.json")
    save_json(summary_path, summary)
    save_task_snapshot(cfg, 4, status="started")
    for split in ("tuning", "evaluation"):
        definition = spatial_dependence_definition(cfg, split)
        activities = load_activity_samples(result_path(out, "activity", f"{split}_activity.npz"))
        folder = ensure_dir(task_dir(out, 4) / split)
        nonzeros = []
        for q, activity in enumerate(tqdm(activities, desc=f"Task 04 {split} spatial matrices",
                                        unit="realization", disable=not show_progress)):
            if len(activity.a) != definition["N"]:
                raise ValueError("Saved activity dimensions differ from configuration; rerun Task 02.")
            C = csr_matrix(build_C(activity.device_locations, rho=definition["rho"], U=definition["U"]))
            save_sparse_npz(folder / f"{q:04d}.npz", C, compressed=True)
            nonzeros.append(C.nnz)
        summary[split] = {"samples": len(activities), "definition": definition,
                          "mean_nonzeros": float(np.mean(nonzeros)),
                          "mean_density": float(np.mean(nonzeros) / definition["N"]**2)}
    summary["status"] = "complete"
    summary["evaluation_shared_between_scenarios"] = True
    save_json(summary_path, summary)
    save_task_snapshot(cfg, 4, status="complete", spatial_dependence_datasets=summary)
    print_task_report("TASK 04 — SPATIAL DEPENDENCE MATRICES", {
        "tuning matrices": summary["tuning"]["samples"],
        "shared evaluation matrices": summary["evaluation"]["samples"],
        "rho (m)": definition["rho"], "U (m)": definition["U"], "format": "compressed sparse CSR",
    }, [task_dir(out, 4) / "tuning", task_dir(out, 4) / "evaluation", summary_path])
    return summary


def _tune_pair(args: tuple[Any, ...]) -> dict[str, Any]:
    activities, communication, matrices, alpha, beta, alg = args
    gammas: list[np.ndarray] = []
    nmse_values: list[float] = []
    iterations: list[int] = []
    for i, activity in enumerate(activities):
        r = communication[i]
        C = matrices[i]
        result = cgap_sbl_anc(
            r.Theta, r.Y, r.sigma2, C, alpha=alpha, beta=beta,
            gamma_init=alg["gamma_init"], K=alg["K"], delta=alg["delta"],
        )
        gammas.append(result.gamma); nmse_values.append(nmse(r.Z, result.mu, activity.a)); iterations.append(result.iterations)
    gammas_arr = np.asarray(gammas)
    a_true = np.stack([a.a for a in activities])
    curve = threshold_curve(a_true, gammas_arr)
    threshold = max(curve, key=lambda x: (x["f1"], x["recall"], x["precision"]))
    return {
        "alpha": float(alpha), "beta": float(beta), "tau": threshold["tau"], "threshold_curve": curve,
        "precision": threshold["precision"], "recall": threshold["recall"], "f1": threshold["f1"],
        "nmse": float(np.nanmean(nmse_values)), "iterations": float(np.mean(iterations)),
    }


def tune_stage(cfg: dict[str, Any], workers: int = 1, show_progress: bool = True) -> dict[str, Any]:
    """Task 05: jointly select alpha, beta and final detection thresholds."""
    out = output_dir(cfg)
    tune = cfg["tuning"]
    if 0 not in tune["alpha_values"] or 0 not in tune["beta_values"]:
        raise ValueError("The tuning grid must include alpha=0 and beta=0 for the SBL baseline.")
    activities = load_activity_samples(result_path(out, "activity", "tuning_activity.npz"))
    communication = load_tuning_communication(cfg, len(activities))
    matrices = load_tuning_spatial_dependence(cfg, len(activities))
    tasks = [(activities, communication, matrices, alpha, beta, cfg["algorithms"])
             for alpha in tune["alpha_values"] for beta in tune["beta_values"]]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(tqdm(ex.map(_tune_pair, tasks), total=len(tasks), desc="Task 05 alpha-beta-tau grid", unit="pair", disable=not show_progress))
    else:
        rows = [_tune_pair(t) for t in tqdm(tasks, desc="Task 05 alpha-beta-tau grid", unit="pair", disable=not show_progress)]
    rows.sort(key=lambda x: (x["f1"], -x["nmse"], x["recall"]), reverse=True)
    best = rows[0]
    baseline = next(row for row in rows if row["alpha"] == 0 and row["beta"] == 0)
    curves = [dict(alpha=row["alpha"], beta=row["beta"], **point)
              for row in rows for point in row["threshold_curve"]]
    artifacts = []
    for filename, data in [("alpha_beta_tau_scores.csv", curves),
                           ("cgap_thresholds.csv", best["threshold_curve"]),
                           ("sbl_thresholds.csv", baseline["threshold_curve"])]:
        path = result_path(out, "tuning", filename)
        save_csv(path, data)
        artifacts.append(path)
    for row in rows:
        row.pop("threshold_curve")
    grid_path = result_path(out, "tuning", "alpha_beta.csv")
    save_csv(grid_path, rows)
    def selection(row):
        return {key: row[key] for key in ("alpha", "beta", "tau")} | {
            "tuning_precision": row["precision"], "tuning_recall": row["recall"],
            "tuning_f1": row["f1"], "tuning_nmse": row["nmse"]}
    selected = {"definition": tuning_definition(cfg), "cgap": selection(best),
                "sbl": selection(baseline), "reference_snr_db": tune["reference_snr_db"],
                "reference_pilot_length": tune["reference_pilot_length"]}
    selected_path = result_path(out, "tuning", "selected.json")
    save_json(selected_path, selected)
    summary_path = result_path(out, "tuning", "alpha_beta_tau_summary.json")
    save_json(summary_path, {"best": selected, "grid_pairs": len(rows), "tau_candidates": 201, "top5": rows[:5]})
    save_task_snapshot(cfg, 5, workers=workers, show_progress=show_progress, selected_parameters=selected)
    print_task_report("TASK 05 — ALPHA/BETA/TAU TUNING", {
        "grid pairs": len(rows), "tau candidates": 201, "selected alpha": best["alpha"],
        "selected beta": best["beta"], "CGAP-SBL-ANC tau": best["tau"],
        "SBL tau": baseline["tau"], "tuning F1": best["f1"], "mean NMSE": best["nmse"],
    }, [grid_path, selected_path, summary_path, *artifacts])
    return selected


def evaluation_conditions(cfg: dict[str, Any]) -> list[tuple[int, float, str]]:
    sys = cfg["system"]; tune = cfg["tuning"]
    L0 = int(cfg["evaluation"]["pilot_length"]); snr0 = float(tune["reference_snr_db"])
    conditions: list[tuple[int, float, str]] = []
    for snr in sys["snr_db"]:
        conditions.append((L0, float(snr), "snr_sweep"))
    for L in sys["pilot_lengths"]:
        conditions.append((int(L), snr0, "pilot_sweep"))
    return conditions


def _evaluate_sample(args: tuple[Any, ...]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray] | None]:
    q, activity, communication_folder, sys, spatial_dependence_folder, alg, selected, conditions, save_gamma = args
    rows = []
    gamma_save = None
    communication = load_communication_sample(communication_folder, q)
    C = load_spatial_dependence_matrix(spatial_dependence_folder, q)
    options = {key: alg[key] for key in ("gamma_init", "K", "delta")}
    for L, snr_db, sweep in conditions:
        paired = {scenario: communication[(scenario, L, float(snr_db))] for scenario in ("correlated", "randomized")}
        for scenario, r in paired.items():
            common = {"q": q, "S_q": int(np.sum(r.a)), "scenario": scenario,
                      "L": L, "snr_db": snr_db, "sweep": sweep, "nmse_scope": NMSE_SCOPE}
            t0 = time.perf_counter()
            cgap = cgap_sbl_anc(r.Theta, r.Y, r.sigma2, C,
                selected["cgap"]["alpha"], selected["cgap"]["beta"], **options)
            cgap_runtime = time.perf_counter() - t0
            t0 = time.perf_counter()
            sb = sbl(r.Theta, r.Y, r.sigma2, **options)
            sbl_runtime = time.perf_counter() - t0
            for name, result, key, runtime in (("CGAP-SBL-ANC", cgap, "cgap", cgap_runtime), ("SBL", sb, "sbl", sbl_runtime)):
                precision, recall, f1 = precision_recall_f1(r.a, support_from_gamma(result.gamma, selected[key]["tau"]))
                rows.append({**common, "algorithm": name, "precision": precision, "recall": recall,
                             "f1": f1, "nmse": nmse(r.Z, result.mu, r.a), "iterations": result.iterations, "runtime_s": runtime})
            if scenario == "randomized":
                continue
            B_q = oracle_sparsity_budget(common["S_q"], L, sys["N"])
            for name, solver in (("MMV-OMP", mmv_omp), ("MMV-CoSaMP", mmv_cosamp)):
                greedy_options = {} if name == "MMV-OMP" else {"max_iter": alg["cosamp_max_iter"], "tol": alg["cosamp_tol"]}
                t0 = time.perf_counter()
                Z_hat = solver(r.Theta, r.Y, B_q, **greedy_options)
                runtime = time.perf_counter() - t0
                precision, recall, f1 = precision_recall_f1(r.a, support_from_rows(Z_hat, B_q))
                rows.append({**common, "algorithm": name, "precision": precision, "recall": recall, "f1": f1,
                             "nmse": nmse(r.Z, Z_hat, r.a), "iterations": B_q if name == "MMV-OMP" else np.nan,
                             "B_q": B_q, "oracle_capped": B_q < common["S_q"], "runtime_s": runtime})
            if save_gamma and L == selected["reference_pilot_length"] and snr_db == selected["reference_snr_db"]:
                gamma_save = {"a": r.a.astype(np.uint8), "cgap_gamma": cgap.gamma, "sbl_gamma": sb.gamma}
    return rows, gamma_save


def evaluate_stage(cfg: dict[str, Any], workers: int = 1, show_progress: bool = True) -> dict[str, Any]:
    """Task 06: run all estimators and save raw per-sample metrics."""
    out = output_dir(cfg); sys = cfg["system"]; dependence = activity_spatial_dependence_parameters(cfg); alg = cfg["algorithms"]; run = cfg["run"]
    activities = load_activity_samples(result_path(out, "activity", "evaluation_activity.npz"))
    randomized_path = result_path(out, "activity", "randomized_evaluation_activity.npz")
    if not randomized_path.exists():
        raise FileNotFoundError(f"Run Task 02 to save randomized evaluation activity: {randomized_path}")
    randomized = load_activity_samples(randomized_path)
    if len(randomized) != len(activities):
        raise ValueError("Task-02 paired dataset lengths differ; rerun Task 02.")
    with (result_path(out, "tuning", "selected.json")).open("r", encoding="utf-8") as f:
        selected = json.load(f)
    validate_selection(selected, cfg)
    selected["reference_snr_db"] = cfg["tuning"]["reference_snr_db"]
    selected["reference_pilot_length"] = cfg["evaluation"]["pilot_length"]
    conditions = evaluation_conditions(cfg)
    n_gamma = min(int(cfg["evaluation"].get("gamma_distribution_samples", 0)), len(activities))
    communication_folder = validate_communication_dataset(cfg, "evaluation")
    spatial_dependence_folder = validate_spatial_dependence_dataset(cfg, "evaluation")
    tasks = [(i, a, communication_folder, sys, spatial_dependence_folder, alg, selected, conditions, i < n_gamma) for i, a in enumerate(activities)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            outputs = list(tqdm(ex.map(_evaluate_sample, tasks), total=len(tasks), desc="Task 06 evaluation", unit="sample", disable=not show_progress))
    else:
        outputs = [_evaluate_sample(t) for t in tqdm(tasks, desc="Task 06 evaluation", unit="sample", disable=not show_progress)]
    rows = [row for sample_rows, _ in outputs for row in sample_rows]
    per_sample_path = result_path(out, "evaluation", "per_sample.csv")
    save_csv(per_sample_path, rows)
    gamma_outputs = [g for _, g in outputs if g is not None]
    gamma_path = result_path(out, "evaluation", "gamma_distribution.npz")
    if gamma_outputs:
        save_npz(gamma_path,
                 a=np.stack([g["a"] for g in gamma_outputs]),
                 cgap_gamma=np.stack([g["cgap_gamma"] for g in gamma_outputs]),
                 sbl_gamma=np.stack([g["sbl_gamma"] for g in gamma_outputs]))
    realized_counts = np.asarray([np.sum(a.a) for a in activities], dtype=float)
    capped_by_L = {}
    for row in rows:
        if row["algorithm"] == "MMV-OMP" and bool(row.get("oracle_capped", False)):
            key = str(int(row["L"]))
            capped_by_L[key] = capped_by_L.get(key, 0) + 1
    summary = {
        "nmse_scope": NMSE_SCOPE, "Q": len(activities), "conditions_per_realization": len(conditions), "scenarios": ["correlated", "randomized"], "metric_rows": len(rows),
        "gamma_samples": len(gamma_outputs), "workers": int(workers),
        "mean_realized_S": float(np.mean(realized_counts)),
        "std_realized_S": float(np.std(realized_counts, ddof=1)) if len(realized_counts) > 1 else 0.0,
        "min_realized_S": int(np.min(realized_counts)), "max_realized_S": int(np.max(realized_counts)),
        "oracle_budget_rule": "B_q = min(S_q, L, N)",
        "timing": {
            "clock": "time.perf_counter", "unit": "seconds", "scope": "one solver call",
            "includes": "initialization, solver validation, all iterations, final channel estimate",
            "excludes": "data/matrix generation, archive loading, support thresholding, metrics, plotting, saving",
            "C_storage": "CSR; constructed in Task 04",
        },
        "oracle_capped_conditions": int(sum(capped_by_L.values())),
        "oracle_capped_by_L": capped_by_L,
    }
    summary_path = result_path(out, "evaluation", "run_summary.json"); save_json(summary_path, summary)
    files: list[Path] = [per_sample_path, summary_path]
    if gamma_outputs: files.append(gamma_path)
    save_task_snapshot(cfg, 6, workers=workers, show_progress=show_progress, selected_parameters=selected, communication_master_seed=int(cfg["seeds"]["communication_evaluation"]))
    print_task_report("TASK 06 — ESTIMATOR RUNS", summary, files)
    aggregate_stage(cfg)
    return summary


def convergence_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    out = output_dir(cfg); sys = cfg["system"]; dependence = activity_spatial_dependence_parameters(cfg); alg = cfg["algorithms"]; run = cfg["run"]; tune = cfg["tuning"]
    activities = load_activity_samples(result_path(out, "activity", "evaluation_activity.npz"))
    idx = int(cfg["evaluation"].get("convergence_sample_offset", 0))
    activity = activities[idx]
    with (result_path(out, "tuning", "selected.json")).open("r", encoding="utf-8") as f:
        selected = json.load(f)
    validate_selection(selected, cfg)
    folder = validate_communication_dataset(cfg, "evaluation")
    r = load_communication_sample(folder, idx)[("correlated", int(cfg["evaluation"]["pilot_length"]), float(tune["reference_snr_db"]))]
    C = load_spatial_dependence_matrix(validate_spatial_dependence_dataset(cfg, "evaluation"), idx)
    print("Task 07: running CGAP-SBL-ANC convergence trace...")
    cgap = cgap_sbl_anc(r.Theta, r.Y, r.sigma2, C, selected["cgap"]["alpha"], selected["cgap"]["beta"], gamma_init=alg["gamma_init"], K=alg["K"], delta=alg["delta"], keep_history=True, Z_true=r.Z)
    print("Task 07: running SBL convergence trace...")
    sb = sbl(r.Theta, r.Y, r.sigma2, gamma_init=alg["gamma_init"], K=alg["K"], delta=alg["delta"], keep_history=True, Z_true=r.Z)
    conv_path = result_path(out, "convergence", "convergence.npz")
    save_npz(conv_path, cgap_gamma_history=cgap.gamma_history, cgap_lambda_history=cgap.lambda_history, cgap_nmse_history=cgap.nmse_history, sbl_gamma_history=sb.gamma_history, sbl_nmse_history=sb.nmse_history)
    summary = {
        "nmse_scope": NMSE_SCOPE, "q": idx, "L": int(cfg["evaluation"]["pilot_length"]),
        "snr_db": float(tune["reference_snr_db"]), "K": alg["K"], "delta": alg["delta"],
        "cgap_final_nmse": nmse(r.Z, cgap.mu, r.a), "sbl_final_nmse": nmse(r.Z, sb.mu, r.a),
        "cgap_iterations": int(cgap.iterations), "cgap_converged": bool(cgap.converged),
        "sbl_iterations": int(sb.iterations), "sbl_converged": bool(sb.converged),
        "cgap_last_e_step_nmse": float(cgap.nmse_history[-1]) if cgap.nmse_history is not None and len(cgap.nmse_history) else None,
        "sbl_last_e_step_nmse": float(sb.nmse_history[-1]) if sb.nmse_history is not None and len(sb.nmse_history) else None,
    }
    summary_path = result_path(out, "convergence", "summary.json"); save_json(summary_path, summary)
    save_task_snapshot(cfg, 7, sample_index=idx, selected_parameters=selected, communication_master_seed=int(cfg["seeds"]["communication_evaluation"]))
    print_task_report("TASK 07 — CONVERGENCE", summary, [conv_path, summary_path])
    return summary


def aggregate_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    """Task 06: aggregate raw estimator metrics; no estimators are rerun."""
    out = output_dir(cfg)
    per_sample_path = result_path(out, "evaluation", "per_sample.csv")
    df = pd.read_csv(per_sample_path)
    if "scenario" not in df or "CGAP-SBL-ANC" not in set(df["algorithm"]):
        raise ValueError("Legacy estimator results cannot be relabeled. Rerun Task 06 for CGAP-SBL-ANC.")
    if "nmse_scope" not in df or not df["nmse_scope"].eq(NMSE_SCOPE).all():
        raise ValueError("Saved NMSE does not use true active support. Rerun Task 06.")
    aggregate = df.groupby(["scenario", "algorithm", "L", "snr_db", "sweep"], as_index=False)[["precision", "recall", "f1", "nmse", "runtime_s"]].mean()
    aggregate["nmse_scope"] = NMSE_SCOPE
    aggregate_path = result_path(out, "evaluation", "aggregate.csv")
    ensure_dir(aggregate_path.parent)
    aggregate.to_csv(aggregate_path, index=False)
    paired = df[df["algorithm"].isin(["CGAP-SBL-ANC", "SBL"])].pivot(
        index=["q", "scenario", "L", "snr_db", "sweep"], columns="algorithm", values=["f1", "nmse"])
    advantages = pd.DataFrame({
        "f1_gain": paired["f1"]["CGAP-SBL-ANC"] - paired["f1"]["SBL"],
        "nmse_reduction": paired["nmse"]["SBL"] - paired["nmse"]["CGAP-SBL-ANC"],
    }).reset_index().groupby(["scenario", "L", "snr_db", "sweep"], as_index=False)[["f1_gain", "nmse_reduction"]].mean()
    advantages_path = result_path(out, "evaluation", "spatial_randomization_advantages.csv")
    advantages.to_csv(advantages_path, index=False)
    overall = df.groupby(["scenario", "algorithm"])[["precision", "recall", "f1", "nmse", "runtime_s"]].mean().reset_index()
    overall_path = result_path(out, "evaluation", "overall_by_algorithm.csv"); overall.to_csv(overall_path, index=False)
    runtime_summary = df.groupby(["scenario", "algorithm", "L", "snr_db", "sweep"])["runtime_s"].agg(
        samples="count", mean_s="mean", std_s="std", median_s="median",
        min_s="min", max_s="max", total_s="sum", p95_s=lambda values: values.quantile(.95),
    ).reset_index()
    runtime_path = result_path(out, "evaluation", "runtime_summary.csv")
    runtime_summary.to_csv(runtime_path, index=False)
    ref_L = int(cfg["evaluation"]["pilot_length"]); ref_snr = float(cfg["tuning"]["reference_snr_db"])
    ref = aggregate[(aggregate["scenario"] == "correlated") & (aggregate["L"] == ref_L) & (aggregate["snr_db"] == ref_snr)].drop_duplicates("algorithm")
    reference_rows = ref[["algorithm", "precision", "recall", "f1", "nmse", "runtime_s"]].to_dict(orient="records")
    summary = {"nmse_scope": NMSE_SCOPE, "aggregate_rows": int(len(aggregate)), "reference_L": ref_L, "reference_snr_db": ref_snr, "reference_results": reference_rows}
    summary_path = result_path(out, "evaluation", "summary.json"); save_json(summary_path, summary)
    print_task_report("TASK 06 — AGGREGATION / ANALYSIS", {
        "aggregate rows": len(aggregate), "reference L": ref_L, "reference SNR": ref_snr,
        "algorithms": ", ".join(sorted(df["algorithm"].unique())),
    }, [aggregate_path, advantages_path, overall_path, runtime_path, summary_path])
    print("Reference-condition results:")
    if not ref.empty:
        print(ref[["algorithm", "f1", "nmse", "runtime_s"]].to_string(index=False))
        print()
    return summary
