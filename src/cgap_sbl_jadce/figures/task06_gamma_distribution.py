"""Task06 gamma distribution figures."""
from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from cgap_sbl_jadce.paths import result_path, figure_path, task_dir
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save


def plot_gamma_distribution(cfg: dict, formats: str = "png") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    data = np.load(result_path(out, "evaluation", "gamma_distribution.npz"))
    a = data["a"].astype(bool)
    manifest_path = task_dir(out, 6) / "manifest.json"
    saved_cfg = json.loads(manifest_path.read_text())["configuration"] if manifest_path.exists() else cfg
    reference = saved_cfg.get("task_runtime", {}).get("selected_parameters", saved_cfg["tuning"])
    settings = fr"$L={reference['reference_pilot_length']:g}$, $\mathrm{{SNR}}={reference['reference_snr_db']:g}$ dB"
    paths: list[Path] = []
    for key, label, stem in [("cgap_gamma", "CGAP-SBL-ANC", "06_cgap_gamma_distribution"), ("sbl_gamma", "SBL", "06_sbl_gamma_distribution")]:
        gamma = data[key]
        figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                        "xtick.labelsize": 10, "ytick.labelsize": 10,
                        "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                        "colorbar.ticksize": 10}
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.hist(gamma[~a], bins=50, alpha=0.55, density=True, label="Inactive MTD")
        ax.hist(gamma[a], bins=50, alpha=0.55, density=True, label="Active MTD")
        ax.set_xlabel(r"$\gamma_i$"); ax.set_ylabel("Density"); ax.set_title(fr"{label} $\gamma$ distribution" + "\n" + settings)
        ax.legend(); ax.grid(True, alpha=0.2); _apply_figure_style(fig, figure_style); fig.tight_layout()
        paths.extend(_save(fig, figure_path(out, stem), formats))
    return paths
