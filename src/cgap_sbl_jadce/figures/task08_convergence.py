"""Task08 convergence figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cgap_sbl_jadce.paths import result_path, figure_path
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save


def plot_convergence(cfg: dict, formats: str = "png") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    conv = np.load(result_path(out, "convergence", "convergence.npz"))
    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.arange(1, len(conv["cgap_nmse_history"]) + 1), conv["cgap_nmse_history"], label="CGAP-SBL-ANC")
    ax.plot(np.arange(1, len(conv["sbl_nmse_history"]) + 1), conv["sbl_nmse_history"], label="SBL")
    ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\mathrm{NMSE}$"); ax.set_yscale("log")
    ax.grid(True, alpha=0.2); ax.legend(); ax.set_title("Convergence at reference condition")
    _apply_figure_style(fig, figure_style); fig.tight_layout()
    return _save(fig, figure_path(out, "08_convergence_nmse"), formats)
