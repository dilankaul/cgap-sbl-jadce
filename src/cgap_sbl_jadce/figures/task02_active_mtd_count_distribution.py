"""Task02 active mtd count distribution figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cgap_sbl_jadce.dataset import load_activity_samples
from cgap_sbl_jadce.paths import result_path, figure_path
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save


def plot_activity_s_distribution(cfg: dict, formats: str = "png") -> list[Path]:
    """Plot the realized support-size distribution using saved Task 02 data."""
    out = Path(cfg["run"]["output_dir"])
    tuning_path = result_path(out, "activity", "tuning_activity.npz")
    evaluation_path = result_path(out, "activity", "evaluation_activity.npz")
    if not tuning_path.exists() or not evaluation_path.exists():
        raise FileNotFoundError("Task 02 tuning/evaluation activity files are required.")
    tuning = load_activity_samples(tuning_path)
    evaluation = load_activity_samples(evaluation_path)
    tuning_s = np.asarray([np.sum(sample.a) for sample in tuning], dtype=int)
    evaluation_s = np.asarray([np.sum(sample.a) for sample in evaluation], dtype=int)
    all_counts = np.concatenate((tuning_s, evaluation_s))
    lo = int(np.min(all_counts)); hi = int(np.max(all_counts))
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
    target = float(cfg["system"]["S"])

    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.hist(tuning_s, bins=bins, alpha=0.55, density=True, label=f"Tuning (mean={np.mean(tuning_s):.2f})")
    ax.hist(evaluation_s, bins=bins, alpha=0.55, density=True, label=f"Evaluation (mean={np.mean(evaluation_s):.2f})")
    ax.axvline(target, linestyle="--", linewidth=1.0, label=fr"Target mean $S={target:g}$")
    ax.set_xlabel(r"Realized active MTD count $S$")
    ax.set_ylabel("Relative frequency")
    ax.set_title(r"Distribution of Active MTD Count $S$")
    ax.grid(True, alpha=0.2)
    ax.legend()
    _apply_figure_style(fig, figure_style); fig.tight_layout()
    return _save(fig, figure_path(out, "02_activity_realized_s_distribution"), formats)
