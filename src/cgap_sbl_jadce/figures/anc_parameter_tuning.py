from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from cgap_sbl_jadce.paths import tuning_path, tuning_figure_path, result_path, figure_path
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save


def plot_alpha_beta_tau_tuning(cfg: dict, formats: str = "png", study: bool = False) -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    df = pd.read_csv(result_path(out, "anc_grid_study", "alpha_beta.csv") if study else tuning_path(cfg, "anc", "alpha_beta.csv"))
    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    paths = []
    for metric, label, title, cmap in [
        ("f1", r"Mean $F_1$", r"CGAP-SBL-ANC $\alpha$-$\beta$ Tuning" + "\n" + r"Mean Detection $F_1$", "viridis"),
        ("nmse", r"Mean $\mathrm{NMSE}$", r"CGAP-SBL-ANC $\alpha$-$\beta$ Tuning" + "\n" + "Mean Channel Estimation NMSE", "magma"),
        ("tau", r"Selected activity threshold $\tau$", r"CGAP-SBL-ANC $\alpha$-$\beta$ Tuning" + "\n" + r"Selected Threshold $\tau$", "OrRd"),
    ]:
        pivot = df.pivot(index="alpha", columns="beta", values=metric).sort_index().sort_index(axis=1)
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        image = ax.imshow(pivot.values, aspect="auto", origin="lower",
                          interpolation="nearest", cmap=cmap,
                          norm=LogNorm() if metric == "nmse" else None)
        ax.set_xticks(np.arange(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)), [f"{x:g}" for x in pivot.index])
        ax.set_xticks(np.arange(len(pivot.columns) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(pivot.index) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(len(pivot.index)):
            for col in range(len(pivot.columns)):
                value = pivot.iat[row, col]
                red, green, blue, _ = image.cmap(image.norm(value))
                brightness = 0.299 * red + 0.587 * green + 0.114 * blue
                color = "black" if brightness > 0.5 else "white"
                ax.text(col, row, f"{value:.4f}", ha="center", va="center",
                        color=color, fontsize=9)
        fig.colorbar(image, ax=ax, label=label)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\alpha$")
        ax.set_title(title)
        _apply_figure_style(fig, figure_style)
        fig.tight_layout()
        stem = (figure_path(out, "anc_grid_study", f"anc_parameter_tuning_{metric}") if study else
                tuning_figure_path(cfg, "anc", f"anc_parameter_tuning_{metric}"))
        paths.extend(_save(fig, stem, formats))
    return paths


def plot_anc_grid_study(cfg, formats='png'):
    return plot_alpha_beta_tau_tuning(cfg, formats=formats, study=True)
