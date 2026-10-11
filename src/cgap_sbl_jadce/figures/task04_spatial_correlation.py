"""Task04 spatial correlation figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cgap_sbl_jadce.paths import result_path, figure_path
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save


def plot_correlation_preview(cfg: dict, formats: str = "png") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    C = np.load(result_path(out, "correlation", "preview_C.npz"))["C"]
    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(C, aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, label=r"$C_{ij}$")
    ax.set_xlabel(r"MTD $j$"); ax.set_ylabel(r"MTD $i$"); ax.set_title(r"Spatial correlation matrix $C$")
    _apply_figure_style(fig, figure_style); fig.tight_layout()
    return _save(fig, figure_path(out, "04_correlation_matrix"), formats)
