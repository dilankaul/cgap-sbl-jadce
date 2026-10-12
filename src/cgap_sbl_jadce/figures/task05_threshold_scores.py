"""Task05 activity threshold tuning figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from cgap_sbl_jadce.paths import figure_path, tuning_condition_name, task_dir
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save


def plot_threshold_tuning(cfg: dict, formats: str = "png",
                          condition: tuple[int, float] | None = None,
                          setups: list[tuple[int, float]] | None = None) -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    if condition is not None:
        if setups is not None:
            raise ValueError("Use condition or setups, not both.")
        setups = [condition]
    folders = ([task_dir(out, 5) / tuning_condition_name(L, snr) for L, snr in dict.fromkeys(setups)]
               if setups is not None else [task_dir(out, 5)])
    paths: list[Path] = []
    for filename, label, stem in [
        ("cgap_thresholds.csv", "CGAP-SBL-ANC", "05_cgap_threshold"),
        ("sbl_thresholds.csv", "SBL", "05_sbl_threshold"),
    ]:
        data = pd.concat([pd.read_csv(folder / filename) for folder in folders], ignore_index=True)
        for (L, snr), df in data.groupby(["L", "snr_db"], sort=True):
            df = df.sort_values("tau").reset_index(drop=True)
            figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                            "xtick.labelsize": 10, "ytick.labelsize": 10,
                            "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                            "colorbar.ticksize": 10}
            fig, ax = plt.subplots(figsize=(4.8, 3.6))
            best = df.sort_values(["f1", "recall", "precision"], ascending=False, kind="stable").iloc[0]
            for metric, score_label in [("f1", r"$F_1$"), ("precision", "Precision"), ("recall", "Recall")]:
                ax.plot(df["tau"], df[metric], label=score_label,
                        marker="o", markevery=[int(best.name)], markersize=6,
                        markeredgecolor="black", markeredgewidth=0.8)
            ax.axvline(best["tau"], linestyle="--", linewidth=1.0, label=fr"Selected $\tau={best['tau']:.3g}$")
            ax.set_xlabel(r"Threshold $\tau$"); ax.set_ylabel("Score"); ax.set_ylim(-0.02, 1.02)
            ax.set_title(label + "\n" + r"Detection Scores vs Threshold $\tau$" + "\n" + fr"$L={L:g}$, $\mathrm{{SNR}}={snr:g}$ dB"); ax.grid(True, alpha=0.2); ax.legend()
            _apply_figure_style(fig, figure_style); fig.tight_layout(); paths.extend(_save(fig, figure_path(out, stem).parent / tuning_condition_name(L, snr) / stem, formats))
    return paths
