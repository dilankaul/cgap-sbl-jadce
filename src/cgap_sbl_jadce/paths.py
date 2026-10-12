"""Shared output layout for the nine scientific tasks."""
from pathlib import Path
import re

TASK_FOLDERS = {
    1: "01_kappa_calibration", 2: "02_activity_realizations",
    3: "03_communication_data", 4: "04_spatial_correlation",
    5: "05_alpha_beta_tuning", 6: "06_threshold_tuning",
    7: "07_estimators", 8: "08_convergence", 9: "09_evaluation",
}


def task_dir(out: str | Path, task: int) -> Path:
    return Path(out) / TASK_FOLDERS[task]


def result_path(out: str | Path, group: str, filename: str) -> Path:
    """Resolve each artifact to its producing task, including shared old groups."""
    fixed = {"activity_calibration": 1, "activity": 2, "communication": 3,
             "correlation": 4, "convergence": 8}
    if group == "tuning":
        task = 5 if filename in {"alpha_beta.csv", "selected_alpha_beta.json", "alpha_beta_summary.json"} else 6
    elif group == "evaluation":
        task = 7 if filename in {"per_sample.csv", "gamma_distribution.npz", "run_summary.json"} else 9
    else:
        task = fixed[group]
    folder = task_dir(out, task)
    if group == "activity" and filename in {"tuning_activity.npz", "evaluation_activity.npz", "randomized_evaluation_activity.npz"}:
        folder /= filename.removesuffix("_activity.npz")
    return folder / filename


def figure_path(out: str | Path, stem: str) -> Path:
    task = int(stem[:2])
    folder = task_dir(out, task) / "figures"
    realization = (re.match(r"02_activity_realization_(tuning|evaluation|randomized_evaluation)_(\d+)", stem)
                   if task == 2 else re.match(r"03_(tuning|evaluation|randomized_evaluation)_.+_(\d+)$", stem) if task == 3 else None)
    if realization:
        split, index = realization.groups()
        split = "evaluation" if split == "randomized_evaluation" else split
        folder /= f"{split}_{int(index):04d}"
    elif task in {5, 6}:
        folder /= "tuning"
    elif task in {7, 8, 9}:
        folder /= "evaluation"
    return folder / stem
