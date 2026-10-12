"""Shared output layout for the scientific tasks."""
from pathlib import Path
import re

TASK_FOLDERS = {
    1: "01_kappa_calibration", 2: "02_activity_realizations",
    3: "03_communication_data", 4: "04_spatial_dependence",
    5: "05_alpha_beta_tau_tuning",
    6: "06_estimators", 7: "07_convergence",
}


def task_dir(out: str | Path, task: int) -> Path:
    return Path(out) / TASK_FOLDERS[task]


def result_path(out: str | Path, group: str, filename: str) -> Path:
    """Resolve each artifact to its producing task, including shared old groups."""
    fixed = {"activity_calibration": 1, "activity": 2, "communication": 3,
             "spatial_dependence": 4, "convergence": 7}
    if group == "tuning":
        task = 5
    elif group == "evaluation":
        task = 6
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
                   if task == 2 else re.match(r"(?:03|04)_(tuning|evaluation|randomized_evaluation)_.+_(\d+)$", stem) if task in {3, 4} else None)
    if realization:
        split, index = realization.groups()
        split = "evaluation" if split == "randomized_evaluation" else split
        folder /= f"{split}_{int(index):04d}"
    elif task == 5:
        folder /= "tuning"
    elif task in {6, 7}:
        folder /= "evaluation"
    return folder / stem


def tuning_condition_name(L: int, snr_db: float) -> str:
    """Shared data/figure folder name for one communication condition."""
    return f"L{int(L):04d}_snr{float(snr_db):g}dB"
