from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from tqdm.auto import tqdm
from cgap_sbl_jadce.paths import result_path, figure_path
from cgap_sbl_jadce.settings import evaluation_settings
from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save, sweep_limits


def _line_plot(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, stem: Path, formats: str, xlim: tuple[float, float]) -> list[Path]:
    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for algorithm, group in df.groupby("algorithm"):
        group = group.sort_values(x)
        ax.plot(group[x], group[y], marker="s" if algorithm.startswith("SBL") else "o",
                label=algorithm)
    ax.set_xlim(xlim)
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


def make_performance_figures(cfg: dict, formats: str = "png", show_progress: bool = True, method: str | None = None) -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    df = pd.read_csv(result_path(out, method or "comparison", "aggregate.csv"))
    task = ESTIMATORS[method].task if method else "comparison"
    prefix = method or "comparison"
    snr = df[df["sweep"] == "snr_sweep"]
    pilot = df[df["sweep"] == "pilot_sweep"]
    jobs = [
        (snr, "snr_db", "f1", r"Activity Detection vs $\mathrm{SNR}$", r"$\mathrm{SNR}$ (dB)", r"$F_1$ score", figure_path(out, task, f"{prefix}_f1_vs_snr")),
        (snr, "snr_db", "nmse", r"Channel Estimation vs $\mathrm{SNR}$", r"$\mathrm{SNR}$ (dB)", r"$\mathrm{NMSE}$", figure_path(out, task, f"{prefix}_nmse_vs_snr")),
        (pilot, "L", "f1", r"Activity Detection vs Pilot Length $L$", r"Pilot length $L$", r"$F_1$ score", figure_path(out, task, f"{prefix}_f1_vs_pilot_length")),
        (pilot, "L", "nmse", r"Channel Estimation vs Pilot Length $L$", r"Pilot length $L$", r"$\mathrm{NMSE}$", figure_path(out, task, f"{prefix}_nmse_vs_pilot_length")),
        (snr, "snr_db", "runtime_s", r"Runtime vs $\mathrm{SNR}$", r"$\mathrm{SNR}$ (dB)", "Mean runtime (s)", figure_path(out, task, f"{prefix}_runtime_vs_snr")),
        (pilot, "L", "runtime_s", r"Runtime vs Pilot Length $L$", r"Pilot length $L$", "Mean runtime (s)", figure_path(out, task, f"{prefix}_runtime_vs_pilot_length")),
    ]
    settings = evaluation_settings(cfg)
    limits = {"snr_db": sweep_limits(settings["snr_values"]),
              "L": sweep_limits(settings["pilot_lengths"])}
    paths: list[Path] = []
    for algorithm, data in df[df.sweep == 'condition'].groupby('algorithm'):
        for metric, cmap in [('f1', 'viridis'), ('nmse', 'magma'), ('runtime_s', 'cividis')]:
            pivot = data.pivot(index='L', columns='snr_db', values=metric).sort_index().sort_index(axis=1)
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            image = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap=cmap,
                              norm=LogNorm() if metric == 'nmse' else None)
            ax.set_xticks(range(len(pivot.columns)), [f'{value:g}' for value in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [f'{value:g}' for value in pivot.index])
            ax.set_xlabel('SNR (dB)'); ax.set_ylabel(r'Pilot length $L$')
            ax.set_title(f'{algorithm} {metric.upper()} by Communication Condition')
            fig.colorbar(image, ax=ax, label='Mean solver runtime (s)' if metric == 'runtime_s' else metric.upper())
            fig.tight_layout()
            paths.extend(_save(fig, figure_path(out, task, f'{prefix}_{algorithm.lower()}_{metric}_condition_grid'), formats))
    jobs = [job for job in jobs if not job[0].empty]
    for job in tqdm(jobs, desc=f"{prefix} performance figures", unit="figure", disable=not show_progress):
        paths.extend(_line_plot(*job, formats=formats, xlim=limits[job[1]]))
    return paths
