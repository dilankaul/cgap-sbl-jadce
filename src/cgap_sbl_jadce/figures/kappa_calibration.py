from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cgap_sbl_jadce.paths import result_path, figure_path
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save, sweep_limits


def plot_activity_calibration(cfg: dict, formats: str = "png") -> list[Path]:
    import json
    out = Path(cfg["run"]["output_dir"])
    path = result_path(out, "kappa_calibration", "kappa_sweep.csv")
    summary_path = result_path(out, "kappa_calibration", "summary.json")
    df = pd.read_csv(path).sort_values("kappa")
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        target = float(summary["target_mean_active"])
        D = float(summary["D"])
    else:
        target = float(cfg["system"]["S"])
        D = float(cfg["activity"]["D"])
    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    ax.plot(
        df["kappa"], df["mean_expected_active"],
        color="#0072B2", linewidth=1.4, marker="o", markersize=6,
        zorder=3,
        label=r"Mean $\sum_i P_i$",
    )
    ax.plot(
        df["kappa"], df["mean_realized_active"],
        color="#D55E00", linestyle="--", linewidth=1.2,
        marker="s", markersize=3, zorder=4,
        label=r"Mean realized active MTD count $S$",
    )
    ax.axhline(target, linestyle="--", linewidth=1.0, label=fr"Target mean $S={target:g}$")
    best = df.iloc[int(np.argmin(np.abs(df["mean_expected_active"].to_numpy() - target)))]
    ax.axvline(best["kappa"], linestyle=":", linewidth=1.0, label=fr"Closest grid $\kappa={best['kappa']:.4g}$")
    ax.set_xlim(sweep_limits(df["kappa"]))
    ax.set_xlabel(r"Decay parameter $\kappa$")
    ax.set_ylabel(r"Average active MTD count $S$")
    ax.set_title(r"Mean Active MTD Count vs Decay Parameter $\kappa$" + "\n" + fr"Event influence radius $D={D:g}$ m")
    ax.grid(True, alpha=0.2); ax.legend()
    _apply_figure_style(fig, figure_style); fig.tight_layout()
    return _save(fig, figure_path(out, "kappa_calibration", "activity_kappa_calibration"), formats)
