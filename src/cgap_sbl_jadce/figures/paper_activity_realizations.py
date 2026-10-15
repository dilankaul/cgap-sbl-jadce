from __future__ import annotations
from pathlib import Path
from cgap_sbl_jadce.dataset import load_activity_samples
from cgap_sbl_jadce.paths import result_path
from cgap_sbl_jadce.figures.activity_realizations import _activity_plot_options, plot_activity_sample


def plot_paper_activity_figure(cfg: dict, split: str, index: int, formats: str = "pdf") -> Path:
    if split not in {"tuning", "evaluation"}:
        raise ValueError("Paper realization split must be tuning or evaluation")
    if type(index) is not int or index < 0:
        raise ValueError(f"Invalid {split} realization index: {index}")
    out = Path(cfg["run"]["output_dir"])
    destination = out / "paper_figures" / "activity"

    figure_style = {"axes.titlesize": 15, "axes.labelsize": 15,
                    "xtick.labelsize": 13, "ytick.labelsize": 13,
                    "legend.fontsize": 13, "colorbar.labelsize": 15,
                    "colorbar.ticksize": 13}
    figsize = (6.8, 6.2)
    samples = load_activity_samples(result_path(out, "activity", f"{split}_activity.npz"))
    if index >= len(samples):
        raise ValueError(f"Invalid {split} realization index: {index}; expected 0 to {len(samples) - 1}")
    return plot_activity_sample(
        samples[index],
        stem=destination / split / f"event_driven_activation_model_{split}_{index:04d}",
        formats=formats, title=None,
        style=figure_style,
        figsize=figsize,
        **_activity_plot_options(cfg),
    )[0]
