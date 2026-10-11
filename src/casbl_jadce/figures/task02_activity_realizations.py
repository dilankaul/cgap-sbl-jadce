"""Task02 activity realizations figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle
import numpy as np
from tqdm.auto import tqdm
from casbl_jadce.models.activity import ActivitySample, event_activation_probabilities
from casbl_jadce.config import resolve_activity_kappa
from casbl_jadce.dataset import load_activity_samples
from casbl_jadce.paths import result_path, figure_path
from casbl_jadce.figures.common import figure_formats, _apply_figure_style, _save


def activity_probability_grid(
    event_locations: np.ndarray,
    R: float,
    D: float,
    kappa: float,
    resolution: int = 301,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """Evaluate the paper's combined activation probability over the cell.

    At every spatial grid point x, the plotted field is exactly
    P(x) = 1 - prod_v(1 - P_v(x)). Points outside the BS cell are masked.
    """
    resolution = max(51, int(resolution))
    x = np.linspace(-float(R), float(R), resolution)
    y = np.linspace(-float(R), float(R), resolution)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel()))
    P = event_activation_probabilities(points, np.asarray(event_locations), kappa=float(kappa), D=float(D)).reshape(X.shape)
    outside = X * X + Y * Y > float(R) ** 2
    return X, Y, np.ma.array(P, mask=outside)



def plot_activity_sample(
    sample: ActivitySample,
    R: float,
    D: float,
    stem: Path,
    formats: str = "png",
    show_event_radius: bool = True,
    title: str | None = None,
    kappa: float | None = None,
    show_probability_field: bool = False,
    probability_resolution: int = 301,
    probability_alpha: float = 0.35,
    probability_levels: int = 128,
    style: dict | None = None,
    figsize: tuple[float, float] = (6.8, 6.2),
    show_events: bool = True,
) -> list[Path]:
    """Plot one activity sample without requiring Task 02 output files."""
    loc = sample.device_locations
    active = sample.a.astype(bool)

    figure_style = {"axes.titlesize": 15, "axes.labelsize": 15,
                    "xtick.labelsize": 13, "ytick.labelsize": 13,
                    "legend.fontsize": 13, "colorbar.labelsize": 15,
                    "colorbar.ticksize": 13}
    figure_style.update(style or {})
    fig, ax = plt.subplots(figsize=figsize)
    if show_probability_field:
        if kappa is None:
            raise ValueError("kappa is required when show_probability_field=True.")
        X, Y, P = activity_probability_grid(
            sample.event_locations, R=R, D=D, kappa=kappa, resolution=probability_resolution
        )

        # Plot the calculated probability field without modifying P. The only
        # visualization mapping is the colormap: P=0 is pure white and P=1 is
        # dark red. The mask in P applies only outside the physical BS cell.
        n_levels = max(8, int(probability_levels))
        levels = np.linspace(0.0, 1.0, n_levels)
        probability_cmap = LinearSegmentedColormap.from_list(
            "activation_probability_white_red",
            [
                (0.00, "#ffffff"),  # Pure white
                (0.25, "#f5b5b5"),  # Visible light red
                (0.50, "#e87c7c"),  # Medium red
                (0.75, "#d44747"),  # Strong red
                (1.00, "#a81818"),  # Deep red
            ],
            N=256,
        )
        probability_norm = Normalize(vmin=0.0, vmax=1.0)
        image = ax.contourf(
            X, Y, P, levels=levels, cmap=probability_cmap, norm=probability_norm,
            alpha=float(probability_alpha), antialiased=True, zorder=0,
        )
        cbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            ticks=np.linspace(0.0, 1.0, 6),
        )
        cbar.set_label(
            r"Activation probability $P_{\mathrm{act}}(\mathbf{r})$"
        )

    if np.any(~active):
        ax.scatter(
            loc[~active, 0], loc[~active, 1],
            marker="o", s=18, alpha=0.75,
            label="Inactive MTD", zorder=3,
            c="lightsteelblue")
    if np.any(active):
        ax.scatter(
            loc[active, 0], loc[active, 1],
            marker="o", s=30, alpha=0.75,
            label="Active MTD", zorder=4,
            c="forestgreen")
    if show_events:
        ax.scatter(
            sample.event_locations[:, 0], sample.event_locations[:, 1],
            marker="*", s=150, c="firebrick", edgecolors="firebrick",
            linewidths=0.6, alpha=1.0, label="Event", zorder=5,
        )
    ax.scatter(
        [0.0], [0.0],
        marker="x", s=90, c="black", alpha=1.0,
        label="BS", zorder=5,
    )
    ax.add_patch(Circle((0.0, 0.0), R, fill=False, edgecolor="#333333", linestyle="--", linewidth=1.2, zorder=1, label="Cell boundary"))
    if show_event_radius:
        first = True
        for xy in sample.event_locations:
            ax.add_patch(Circle(tuple(xy), D, fill=False, edgecolor="#888888", linestyle="--", linewidth=0.8, alpha=0.75, zorder=2, label="Event influence boundary" if first else None))
            first = False
    ax.set_aspect("equal", adjustable="box")
    pad = 1.08 * R
    ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad)
    ax.set_xlabel(r"$x$ (m)"); ax.set_ylabel(r"$y$ (m)")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.2); ax.legend(loc="best", facecolor="white", framealpha=1.0)
    _apply_figure_style(fig, figure_style); fig.tight_layout()
    return _save(fig, stem, formats)



def _saved_activity_parameters(cfg: dict) -> tuple[float, float, float]:
    """Use parameters recorded by Task 02 when available.

    This prevents a misleading plot if the config is edited after activity data
    were generated. Old files without a summary fall back to the config.
    """
    out = Path(cfg["run"]["output_dir"])
    summary_path = result_path(out, "activity", "summary.json")
    if summary_path.exists():
        import json
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        return (
            float(summary.get("R", cfg["system"]["R"])),
            float(summary.get("D", cfg["activity"]["D"])),
            float(summary["kappa"]) if "kappa" in summary else resolve_activity_kappa(cfg)[0],
        )
    return float(cfg["system"]["R"]), float(cfg["activity"]["D"]), resolve_activity_kappa(cfg)[0]



def _activity_plot_options(
    cfg: dict, show_event_radius: bool | None = None,
    show_probability_field: bool | None = None,
) -> dict:
    """Resolve shared model/field settings once per activity export."""
    R, D, kappa = _saved_activity_parameters(cfg)
    settings = cfg.get("figures", {})
    return {
        "R": R, "D": D, "kappa": kappa,
        "show_event_radius": settings.get("show_event_radius", True) if show_event_radius is None else show_event_radius,
        "show_probability_field": bool(settings.get("show_probability_field", False)) if show_probability_field is None else show_probability_field,
        "probability_resolution": int(settings.get("probability_resolution", 301)),
        "probability_alpha": float(settings.get("probability_alpha", 0.35)),
        "probability_levels": int(settings.get("probability_levels", 128)),
    }



def plot_activity_realization(
    cfg: dict,
    split: str = "tuning",
    sample_index: int = 0,
    formats: str = "png",
    show_event_radius: bool = True,
    show_probability_field: bool | None = None,
) -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    if split not in {"tuning", "evaluation", "randomized_evaluation"}:
        raise ValueError("split must be tuning, evaluation, or randomized_evaluation.")
    samples = load_activity_samples(result_path(out, "activity", f"{split}_activity.npz"))
    if not 0 <= sample_index < len(samples):
        raise IndexError(f"sample_index must be in [0, {len(samples)-1}].")
    return _plot_activity_realization_sample(
        cfg, samples[sample_index], split, sample_index, formats,
        _activity_plot_options(cfg, show_event_radius, show_probability_field),
    )



def _plot_activity_realization_sample(
    cfg: dict, sample: ActivitySample, split: str, sample_index: int,
    formats: str, options: dict,
) -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    if split == "randomized_evaluation":
        options = {**options, "show_probability_field": False, "show_event_radius": False,
                   "show_events": False}
    suffix = "_probability_field" if options["show_probability_field"] else ""
    return plot_activity_sample(
        sample,
        stem=figure_path(out, f"02_activity_realization_{split}_{sample_index:04d}{suffix}"),
        formats=formats,
        title=f"{split.replace('_', ' ').capitalize()} activity realization {sample_index}: {int(np.sum(sample.a))} active MTDs",
        **options,
    )



def plot_activity_realizations(
    cfg: dict, formats: str = "png", show_progress: bool = True,
    show_probability_field: bool | None = None,
    splits: tuple[str, ...] = ("tuning", "evaluation", "randomized_evaluation"),
) -> list[Path]:
    """Save every tuning and evaluation realization from Task 02 data."""
    if not figure_formats(formats):
        return []
    out = Path(cfg["run"]["output_dir"])
    paths: list[Path] = []
    options = None
    if any(split not in {"tuning", "evaluation", "randomized_evaluation"} for split in splits):
        raise ValueError("Activity figure splits must be tuning, evaluation, or randomized_evaluation")
    for split in splits:
        data_path = result_path(out, "activity", f"{split}_activity.npz")
        if not data_path.exists():
            continue
        if options is None:
            options = _activity_plot_options(cfg, show_probability_field=show_probability_field)
        samples = load_activity_samples(data_path)
        for index, sample in enumerate(tqdm(
            samples, desc=f"Activity figures ({split})", unit="realization",
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            disable=not show_progress,
        )):
            paths.extend(_plot_activity_realization_sample(
                cfg, sample, split, index, formats,
                options,
            ))
    return paths
