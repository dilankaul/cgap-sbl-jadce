from __future__ import annotations

from cgap_sbl_jadce.labels import display_label
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from cgap_sbl_jadce.paths import result_path, figure_path
from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save


def plot_gamma_distribution(cfg: dict, method: str, formats: str = "png") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    spec = ESTIMATORS[method]
    path = result_path(out, method, "gamma_distribution.npz")
    if not path.exists():
        return []
    data = np.load(path)
    a, gamma = data["a"].astype(bool), data["gamma"]
    summary = json.loads(result_path(out, method, "run_summary.json").read_text())
    if summary.get('gamma_samples', 0) == 0:
        return []
    settings = fr"$L={summary['reference_pilot_length']:g}$, $\mathrm{{SNR}}={summary['reference_snr_db']:g}$ dB"
    label, stem = display_label(spec.name), f"{method}_gamma_distribution"
    paths: list[Path] = []
    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(gamma[~a], bins=50, alpha=0.55, density=True, label="Inactive MTD")
    ax.hist(gamma[a], bins=50, alpha=0.55, density=True, label="Active MTD")
    ax.set_xlabel(r"$\gamma_i$"); ax.set_ylabel("Density"); ax.set_title(fr"{label} $\gamma$ distribution" + "\n" + settings)
    ax.legend(); ax.grid(True, alpha=0.2); _apply_figure_style(fig, figure_style); fig.tight_layout()
    paths.extend(_save(fig, figure_path(out, method, stem, section="evaluation"), formats))
    return paths
