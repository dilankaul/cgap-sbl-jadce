"""Common command-line orchestration for the scientific task wrappers."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from collections.abc import Callable

from cgap_sbl_jadce.experiments import pipeline
from cgap_sbl_jadce.figures.task01_kappa_calibration import plot_activity_calibration
from cgap_sbl_jadce.figures.task02_activity_realizations import plot_activity_realizations
from cgap_sbl_jadce.figures.task02_active_mtd_count_distribution import plot_activity_s_distribution
from cgap_sbl_jadce.figures.task03_communication_preview import plot_communication_preview
from cgap_sbl_jadce.figures.task04_spatial_dependence import plot_spatial_dependence
from cgap_sbl_jadce.figures.task05_alpha_beta_tau_tuning import plot_alpha_beta_tau_tuning
from cgap_sbl_jadce.figures.task05_threshold_scores import plot_threshold_tuning
from cgap_sbl_jadce.figures.task06_gamma_distribution import plot_gamma_distribution
from cgap_sbl_jadce.figures.task07_convergence import plot_convergence
from cgap_sbl_jadce.figures.task06_performance import make_performance_figures
from cgap_sbl_jadce.experiments.calibration import CalibrationBracketError
from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.paths import task_dir


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
    3: Task("generate full communication datasets", pipeline.communication_stage, (plot_communication_preview,), True),
    4: Task("build spatial dependence", pipeline.spatial_dependence_stage, (plot_spatial_dependence,), True),
    5: Task("tune alpha, beta and tau", pipeline.tune_stage, (plot_alpha_beta_tau_tuning, plot_threshold_tuning), True),
    6: Task("run estimators and generate performance figures", pipeline.evaluate_stage,
            (plot_gamma_distribution, make_performance_figures), True),
    7: Task("analyze convergence", pipeline.convergence_stage, (plot_convergence,)),
}



def _setup_argument(value: str) -> tuple[int, float]:
    try:
        L, snr = value.split(":")
        setup = (int(L), float(snr))
        if setup[0] <= 0 or not math.isfinite(setup[1]):
            raise ValueError
        return setup
    except ValueError:
        raise argparse.ArgumentTypeError("Use L:SNR, for example 25:8 (positive integer L, finite SNR).") from None


def task_parser(task_number: int) -> argparse.ArgumentParser:
    task = TASKS[task_number]
    parser = argparse.ArgumentParser(description=f"Task {task_number:02d}: {task.description}.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--no-progress", action="store_true")
    if task_number == 5:
        parser.add_argument("--setups", nargs="+", type=_setup_argument, metavar="L:SNR",
                            help="Tune selected setups, e.g. 25:8 30:12; default: all configured setups.")
    return parser


def run_task(task_number: int, argv: list[str] | None = None) -> list:
    """Load effective settings, run one stage, then render its saved outputs."""
    task = TASKS[task_number]
    parser = task_parser(task_number)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg["task_runtime"] = {"input_config": str(args.config), "no_progress": args.no_progress}
    kwargs = {"show_progress": not args.no_progress} if task.progress else {}
    if task_number == 5 and args.setups is not None:
        kwargs["setups"] = args.setups
        cfg["task_runtime"]["setups"] = [list(setup) for setup in args.setups]
    try:
        task.stage(cfg, **kwargs)
    except (CalibrationBracketError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    paths = []
    for render in task.plots:
        kwargs = {"formats": "png"}
        if task_number == 5 and args.setups is not None:
            kwargs["setups"] = args.setups
        if render in (plot_activity_realizations, plot_communication_preview, plot_spatial_dependence, make_performance_figures):
            kwargs["show_progress"] = not args.no_progress
        paths.extend(render(cfg, **kwargs))
    counts = []
    for split in ("tuning", "evaluation"):
        folders = {path.parent for path in paths
                   if path.parent.name.startswith(f"{split}_")
                   and path.parent.name.removeprefix(f"{split}_").isdigit()}
        if folders:
            counts.append(f"{len(folders)} {split} realizations")
    detail = f" ({', '.join(counts)})" if counts else ""
    print(f"Task {task_number:02d} figures: {len(set(paths))} files{detail}")
    if paths:
        print(f"  - {task_dir(cfg['run']['output_dir'], task_number) / 'figures'}")
    return paths
