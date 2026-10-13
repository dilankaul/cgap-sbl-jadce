"""Task06 performance figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from cgap_sbl_jadce.paths import result_path, figure_path
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save


def _line_plot(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, stem: Path, formats: str) -> list[Path]:
    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for algorithm, group in df.groupby("algorithm"):
        group = group.sort_values(x)
        paired = "scenario" in group and len(group["scenario"].unique()) == 1
        randomized = paired and group["scenario"].iloc[0] == "randomized"
        ax.plot(group[x], group[y], marker="s" if algorithm.startswith("SBL") else "o",
                linestyle="--" if randomized else "-", label=algorithm)
    if y == "nmse":
        ax.set_yscale("log")
    fixed_column = "L" if x == "snr_db" else "snr_db"
    fixed_values = df[fixed_column].unique()
    if len(fixed_values) != 1:
        raise ValueError(f"Expected one fixed {fixed_column} value for {stem.name}.")
    fixed = (fr"Fixed $L={fixed_values[0]:g}$" if x == "snr_db"
             else fr"Fixed $\mathrm{{SNR}}={fixed_values[0]:g}$ dB")
    ax.set_title(title + "\n" + fixed); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.25); ax.legend()
    _apply_figure_style(fig, figure_style); fig.tight_layout(); return _save(fig, stem, formats)



def make_performance_figures(cfg: dict, formats: str = "png", show_progress: bool = True) -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    df = pd.read_csv(result_path(out, "evaluation", "aggregate.csv"))
    correlated = df[df["scenario"] == "correlated"]
    snr = correlated[correlated["sweep"] == "snr_sweep"]
    pilot = correlated[correlated["sweep"] == "pilot_sweep"]
    jobs = [
        (snr, "snr_db", "f1", r"Activity Detection vs $\mathrm{SNR}$", r"$\mathrm{SNR}$ (dB)", r"$F_1$ score", figure_path(out, "06_f1_vs_snr")),
        (snr, "snr_db", "nmse", r"Channel Estimation vs $\mathrm{SNR}$", r"$\mathrm{SNR}$ (dB)", r"$\mathrm{NMSE}$", figure_path(out, "06_nmse_vs_snr")),
        (pilot, "L", "f1", r"Activity Detection vs Pilot Length $L$", r"Pilot length $L$", r"$F_1$ score", figure_path(out, "06_f1_vs_pilot_length")),
        (pilot, "L", "nmse", r"Channel Estimation vs Pilot Length $L$", r"Pilot length $L$", r"$\mathrm{NMSE}$", figure_path(out, "06_nmse_vs_pilot_length")),
        (snr, "snr_db", "runtime_s", r"Runtime vs $\mathrm{SNR}$", r"$\mathrm{SNR}$ (dB)", "Mean runtime (s)", figure_path(out, "06_runtime_vs_snr")),
        (pilot, "L", "runtime_s", r"Runtime vs Pilot Length $L$", r"Pilot length $L$", "Mean runtime (s)", figure_path(out, "06_runtime_vs_pilot_length")),
    ]
    comparison = df[df["algorithm"].isin(["CGAP-SBL-ANC", "SBL"])].copy()
    comparison["algorithm"] = comparison["algorithm"] + " (" + comparison["scenario"] + ")"
    for sweep, x, xlabel in [("snr_sweep", "snr_db", r"$\mathrm{SNR}$ (dB)"),
                             ("pilot_sweep", "L", r"Pilot length $L$")]:
        for metric, ylabel in [("f1", r"$F_1$ score"), ("nmse", r"$\mathrm{NMSE}$")]:
            jobs.append((comparison[comparison["sweep"] == sweep], x, metric,
                         "Effect of spatial activity dependence", xlabel, ylabel,
                         figure_path(out, f"06_spatial_dependence_{metric}_{sweep}")))
    paths: list[Path] = []
    for job in tqdm(jobs, desc="Performance figures", unit="figure", disable=not show_progress):
        paths.extend(_line_plot(*job, formats=formats))
    return paths
