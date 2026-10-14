from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cgap_sbl_jadce.paths import result_path, figure_path
from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save


def plot_convergence(cfg: dict, formats: str = "png") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    conv = np.load(result_path(out, "convergence", "convergence.npz"))
    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method, spec in ESTIMATORS.items():
        if f'{method}_nmse_history' in conv:
            history = conv[f'{method}_nmse_history']
            iterations = conv[f'{method}_iteration_history'] if f'{method}_iteration_history' in conv else np.arange(1, len(history) + 1)
            ax.plot(iterations, history, label=spec.name)
    ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\mathrm{NMSE}$"); ax.set_yscale("log")
    ax.grid(True, alpha=0.2); ax.legend(); ax.set_title("Convergence at reference condition")
    _apply_figure_style(fig, figure_style); fig.tight_layout()
    return _save(fig, figure_path(out, "convergence", "convergence_nmse", section="evaluation"), formats)
