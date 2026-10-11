"""Task06 activity threshold tuning figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cgap_sbl_jadce.paths import result_path, figure_path
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save


def plot_threshold_tuning(cfg: dict, formats: str = "png") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    paths: list[Path] = []
    for filename, label, stem in [
        ("cgap_thresholds.csv", "CGAP-SBL-ANC", "06_cgap_threshold"),
        ("sbl_thresholds.csv", "SBL", "06_sbl_threshold"),
    ]:
        df = pd.read_csv(result_path(out, "tuning", filename)).sort_values("tau")
        figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                        "xtick.labelsize": 10, "ytick.labelsize": 10,
                        "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                        "colorbar.ticksize": 10}
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(df["tau"], df["f1"], marker="o", markersize=3, label=r"$F_1$")
        ax.plot(df["tau"], df["precision"], label="Precision")
        ax.plot(df["tau"], df["recall"], label="Recall")
        best = df.iloc[int(np.argmax(df["f1"].to_numpy()))]
        ax.axvline(best["tau"], linestyle="--", linewidth=1.0, label=fr"Selected $\tau={best['tau']:.3g}$")
        ax.set_xlabel(r"Threshold $\tau$"); ax.set_ylabel("Score"); ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{label} threshold tuning"); ax.grid(True, alpha=0.2); ax.legend()
        _apply_figure_style(fig, figure_style); fig.tight_layout(); paths.extend(_save(fig, figure_path(out, stem), formats))
    return paths
