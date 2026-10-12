"""Task05 alpha beta tau tuning figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cgap_sbl_jadce.paths import figure_path, tuning_condition_name, task_dir
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save


def plot_alpha_beta_tau_tuning(cfg: dict, formats: str = "png",
                          condition: tuple[int, float] | None = None,
                          setups: list[tuple[int, float]] | None = None) -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    if condition is not None:
        if setups is not None:
            raise ValueError("Use condition or setups, not both.")
        setups = [condition]
    folders = ([task_dir(out, 5) / tuning_condition_name(L, snr) for L, snr in dict.fromkeys(setups)]
               if setups is not None else [task_dir(out, 5)])
    df = pd.concat([pd.read_csv(folder / "alpha_beta.csv") for folder in folders], ignore_index=True)
    if "nmse_scope" not in df or not df["nmse_scope"].eq("true_active_support").all():
        raise ValueError("Saved tuning scores use the previous NMSE definition. Rerun Task 05.")
    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    paths = []
    for (L, snr), condition_df in df.groupby(["L", "snr_db"], sort=True):
        for metric, label, title, cmap in [
            ("f1", r"Mean $F_1$", r"CGAP-SBL-ANC $\alpha$-$\beta$ Tuning" + "\n" + r"Mean Detection $F_1$", "viridis"),
            ("nmse", r"Mean $\mathrm{NMSE}$ (true active support)", r"CGAP-SBL-ANC $\alpha$-$\beta$ Tuning" + "\n" + "Mean NMSE on True Active Support", "magma"),
            ("tau", r"Selected activity threshold $\tau$", r"CGAP-SBL-ANC $\alpha$-$\beta$ Tuning" + "\n" + r"Selected Threshold $\tau$", "OrRd"),
        ]:
            pivot = condition_df.pivot(index="alpha", columns="beta", values=metric).sort_index().sort_index(axis=1)
            fig, ax = plt.subplots(figsize=(7.0, 5.2))
            image = ax.imshow(pivot.values, aspect="auto", origin="lower",
                              interpolation="nearest", cmap=cmap)
            ax.set_xticks(np.arange(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)), [f"{x:g}" for x in pivot.index])
            ax.set_xticks(np.arange(len(pivot.columns) + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(pivot.index) + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False)
            for row in range(len(pivot.index)):
                for col in range(len(pivot.columns)):
                    value = pivot.iat[row, col]
                    red, green, blue, _ = image.cmap(image.norm(value))
                    brightness = 0.299 * red + 0.587 * green + 0.114 * blue
                    color = "black" if brightness > 0.5 else "white"
                    ax.text(col, row, f"{value:.4f}", ha="center", va="center",
                            color=color, fontsize=9)
            fig.colorbar(image, ax=ax, label=label)
            ax.set_xlabel(r"$\beta$")
            ax.set_ylabel(r"$\alpha$")
            ax.set_title(title + "\n" + fr"$L={L:g}$, $\mathrm{{SNR}}={snr:g}$ dB")
            _apply_figure_style(fig, figure_style)
            fig.tight_layout()
            stem = f"05_alpha_beta_{metric}"
            path = figure_path(out, stem).parent / tuning_condition_name(L, snr) / stem
            paths.extend(_save(fig, path, formats))
    return paths
