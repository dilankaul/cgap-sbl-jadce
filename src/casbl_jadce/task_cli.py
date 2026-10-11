"""Common command-line orchestration for the nine scientific task wrappers."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections.abc import Callable

from casbl_jadce.experiments import pipeline
from casbl_jadce.figures.task01_kappa_calibration import plot_activity_calibration
from casbl_jadce.figures.task02_activity_realizations import plot_activity_realizations
from casbl_jadce.figures.task02_active_mtd_count_distribution import plot_activity_s_distribution
from casbl_jadce.figures.task03_communication_preview import plot_communication_preview
from casbl_jadce.figures.task04_spatial_correlation import plot_correlation_preview
from casbl_jadce.figures.task05_alpha_beta_tuning import plot_alpha_beta_tuning
from casbl_jadce.figures.task06_activity_threshold_tuning import plot_threshold_tuning
from casbl_jadce.figures.task07_gamma_distribution import plot_gamma_distribution
from casbl_jadce.figures.task08_convergence import plot_convergence
from casbl_jadce.figures.task09_performance import make_performance_figures
from casbl_jadce.experiments.calibration import CalibrationBracketError
from casbl_jadce.config import load_config


@dataclass(frozen=True)
class Task:
    description: str
    stage: Callable
    plots: tuple[Callable, ...]
    progress: bool = False


TASKS = {
    1: Task("calibrate kappa", pipeline.calibrate_kappa_stage, (plot_activity_calibration,), True),
    2: Task("generate activity realizations", pipeline.generate_activity_stage,
            (plot_activity_realizations, plot_activity_s_distribution), True),
    3: Task("generate communication preview", pipeline.communication_stage, (plot_communication_preview,)),
    4: Task("build spatial correlation", pipeline.correlation_stage, (plot_correlation_preview,)),
    5: Task("tune CGAP-SBL alpha and beta", pipeline.tune_stage, (plot_alpha_beta_tuning,), True),
    6: Task("tune activity thresholds", pipeline.threshold_stage, (plot_threshold_tuning,), True),
    7: Task("run estimators", pipeline.evaluate_stage, (plot_gamma_distribution,), True),
    8: Task("analyze convergence", pipeline.convergence_stage, (plot_convergence,)),
    9: Task("aggregate and evaluate", pipeline.aggregate_stage, (make_performance_figures,)),
}


def task_parser(task_number: int) -> argparse.ArgumentParser:
    task = TASKS[task_number]
    parser = argparse.ArgumentParser(description=f"Task {task_number:02d}: {task.description}.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def run_task(task_number: int, argv: list[str] | None = None) -> list:
    """Load effective settings, run one stage, then render its saved outputs."""
    task = TASKS[task_number]
    parser = task_parser(task_number)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg["task_runtime"] = {"input_config": str(args.config), "no_progress": args.no_progress}
    kwargs = {"show_progress": not args.no_progress} if task.progress else {}
    try:
        task.stage(cfg, **kwargs)
    except CalibrationBracketError as error:
        parser.exit(1, f"{error}\n")
    paths = []
    for render in task.plots:
        kwargs = {"formats": "png"}
        if render in (plot_activity_realizations, make_performance_figures):
            kwargs["show_progress"] = not args.no_progress
        paths.extend(render(cfg, **kwargs))
    print(f"Task {task_number:02d} figures:")
    reported_paths = dict.fromkeys(
        path.parent if task_number == 2 and path.parent.name in {"tuning", "evaluation", "randomized_evaluation"} else path
        for path in paths
    )
    for path in reported_paths:
        print(f"  - {path}")
    return paths
