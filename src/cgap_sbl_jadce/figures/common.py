from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.text import Text

from cgap_sbl_jadce.io import ensure_dir, atomic_path
from cgap_sbl_jadce.execution_control import pause_point


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Computer Modern Roman",
        "CMU Serif",
        "STIXGeneral",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10.5,
    "axes.linewidth": 0.8,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
})


def figure_formats(mode: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(mode, str):
        mode = mode.lower()
        if mode == "none": return ()
        if mode == "both": return ("png", "pdf")
        if mode in {"png", "pdf"}: return (mode,)
        raise ValueError("figure format must be none, png, pdf, or both")
    return tuple(mode)


def _apply_figure_style(fig: plt.Figure, style: dict) -> None:
    allowed = {"font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
               "ytick.labelsize", "legend.fontsize", "colorbar.labelsize", "colorbar.ticksize"}
    unknown = set(style) - allowed
    if unknown:
        raise ValueError(f"Unsupported figure style settings: {sorted(unknown)}")
    if "font.size" in style:
        for text in fig.findobj(Text):
            text.set_fontsize(style["font.size"])
    for ax in fig.axes:
        for title in (ax.title, ax._left_title, ax._right_title):
            if "axes.titlesize" in style:
                title.set_fontsize(style["axes.titlesize"])
        for label in (ax.xaxis.label, ax.yaxis.label):
            if "axes.labelsize" in style:
                label.set_fontsize(style["axes.labelsize"])
        for axis, key in (("x", "xtick.labelsize"), ("y", "ytick.labelsize")):
            if key in style:
                ax.tick_params(axis=axis, which="both", labelsize=style[key])
                getattr(ax, f"{axis}axis").get_offset_text().set_fontsize(style[key])
        legend = ax.get_legend()
        if legend is not None and "legend.fontsize" in style:
            for text in [*legend.get_texts(), legend.get_title()]:
                text.set_fontsize(style["legend.fontsize"])
        if getattr(ax, "_colorbar", None) is not None:
            if "colorbar.labelsize" in style:
                ax.xaxis.label.set_fontsize(style["colorbar.labelsize"])
                ax.yaxis.label.set_fontsize(style["colorbar.labelsize"])
            if "colorbar.ticksize" in style:
                ax.tick_params(axis="both", which="both", labelsize=style["colorbar.ticksize"])


class _FontTimestampFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in {
            "'created' timestamp seems very low; regarding as unix timestamp",
            "'modified' timestamp seems very low; regarding as unix timestamp",
        }


@contextmanager
def _quiet_font_timestamps():
    logger = logging.getLogger("fontTools.ttLib.tables._h_e_a_d")
    warning_filter = _FontTimestampFilter()
    logger.addFilter(warning_filter)
    try:
        yield
    finally:
        logger.removeFilter(warning_filter)


_destination = ContextVar("figure_destination", default=None)


@contextmanager
def figure_destination(source: Path, destination: Path):
    token = _destination.set((Path(source).resolve(), Path(destination).resolve()))
    try:
        yield
    finally:
        _destination.reset(token)


def _save(fig: plt.Figure, stem: Path, formats: str | Iterable[str] = "pdf") -> list[Path]:
    paths: list[Path] = []
    destination = _destination.get()
    if destination:
        source, folder = destination
        stem = folder / Path(stem).resolve().relative_to(source)
    ensure_dir(stem.parent)
    for ext in figure_formats(formats):
        path = Path(f"{stem}.{ext}")
        kwargs = {"bbox_inches": "tight"}
        if ext == "png": kwargs["dpi"] = 250
        with _quiet_font_timestamps(), atomic_path(path) as temporary:
            fig.savefig(temporary, **kwargs)
        paths.append(path)
    plt.close(fig)
    pause_point()
    return paths


def selected_realizations(cfg: dict, split: str, count: int) -> list[int]:
    indices = cfg.get('figures', {}).get(f'{split}_indices', [0])
    if indices == 'all':
        return list(range(count))
    if not isinstance(indices, list) or any(type(index) is not int for index in indices):
        raise ValueError(f"figures.{split}_indices must be 'all' or a list of integer realization indices.")
    if any(index < 0 or index >= count for index in indices):
        raise ValueError(f'figures.{split}_indices must be within 0..{count - 1}.')
    return list(dict.fromkeys(indices))


def sweep_limits(values):
    low, high = float(min(values)), float(max(values))
    return (low, high) if low < high else (low - .5, high + .5)
