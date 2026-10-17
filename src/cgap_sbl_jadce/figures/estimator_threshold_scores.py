from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from cgap_sbl_jadce.paths import tuning_path, tuning_figure_path
from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save, sweep_limits


def plot_threshold_tuning(cfg: dict, method: str, formats: str = "png") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    paths: list[Path] = []
    spec = ESTIMATORS[method]
    from cgap_sbl_jadce.figures.estimator_parameter_sweeps import selected_parameter_label
    context = selected_parameter_label(cfg, method).replace(' · ', '\n')
    label = spec.name
    stem = f"{method}_threshold_scores"
    df = pd.read_csv(tuning_path(cfg, method, "thresholds.csv")).sort_values("tau").reset_index(drop=True)
    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    best = df.sort_values(["f1", "recall", "precision"], ascending=False, kind="stable").iloc[0]
    for metric, score_label in [("f1", r"$F_1$"), ("precision", "Precision"), ("recall", "Recall")]:
        ax.plot(df["tau"], df[metric], label=score_label,
                marker="o", markevery=[int(best.name)], markersize=6,
                markeredgecolor="black", markeredgewidth=0.8)
    ax.axvline(best["tau"], linestyle="--", linewidth=1.0, label=fr"Selected $\tau={best['tau']:.3g}$")
    ax.set_xlim(sweep_limits(df["tau"]))
    ax.set_xlabel(r"Threshold $\tau$"); ax.set_ylabel("Score"); ax.set_ylim(-0.02, 1.02)
    ax.set_title(label + "\n" + r"Detection Scores vs Threshold $\tau$" + ("\n" + context if context else "")); ax.grid(True, alpha=0.2); ax.legend()
    _apply_figure_style(fig, figure_style); fig.tight_layout(); paths.extend(_save(fig, tuning_figure_path(cfg, method, stem), formats))
    return paths
