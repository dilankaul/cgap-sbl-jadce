"""Task03 communication preview figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cgap_sbl_jadce.paths import result_path, figure_path
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save


def plot_communication_preview(cfg: dict, formats: str = "png") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    data = np.load(result_path(out, "communication", "preview.npz"))
    Theta = data["Theta"]
    gram = np.abs(Theta.conj().T @ Theta)
    np.fill_diagonal(gram, 0.0)
    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(gram, aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, label=r"$|\theta_i^H\theta_j|$")
    ax.set_xlabel(r"Pilot index $j$"); ax.set_ylabel(r"Pilot index $i$")
    ax.set_title("Pilot cross-correlation magnitude (diagonal removed)")
    _apply_figure_style(fig, figure_style); fig.tight_layout()
    return _save(fig, figure_path(out, "03_pilot_cross_correlation"), formats)
