from __future__ import annotations

from cgap_sbl_jadce.labels import display_label
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from cgap_sbl_jadce.paths import tuning_path, tuning_figure_path
from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save, sweep_limits


def plot_threshold_tuning(cfg: dict, method: str, formats: str = "png") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    paths: list[Path] = []
    spec = ESTIMATORS[method]
    from cgap_sbl_jadce.figures.estimator_parameter_sweeps import selected_parameter_label
    context = selected_parameter_label(cfg, method).replace(' · ', '\n').replace('κ_g', r'$\kappa_{\mathrm{g}}$')
    label = display_label(spec.name)
    stem = f"{method}_threshold_scores"
    sources = [('thresholds.csv', '', r'Threshold $\tau$')]
    if tuning_path(cfg, method, 'row_norm_thresholds.csv').exists():
        sources.append(('row_norm_thresholds.csv', '_row_norm', r'Row norm threshold $\tau_Z$'))
    for filename, suffix, xlabel in sources:
        symbol = r'\tau_Z' if suffix else r'\tau'
        df = pd.read_csv(tuning_path(cfg, method, filename)).sort_values("tau").reset_index(drop=True)
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
        ax.axvline(best["tau"], linestyle="--", linewidth=1.0, label=fr"Best diagnostic ${symbol}={best['tau']:.3g}$" if suffix else fr"Selected ${symbol}={best['tau']:.3g}$")
        ax.set_xlim(sweep_limits(df["tau"]))
        ax.set_xlabel(xlabel); ax.set_ylabel("Score"); ax.set_ylim(-0.02, 1.02)
        ax.set_title(label + "\n" + fr"Detection Scores vs Threshold ${symbol}$" + ("\n" + context if context else "")); ax.grid(True, alpha=0.2); ax.legend()
        _apply_figure_style(fig, figure_style); fig.tight_layout(); paths.extend(_save(fig, tuning_figure_path(cfg, method, stem + suffix), formats))
    import numpy as np
    source = tuning_path(cfg, method, 'selected_channel_energy.npz')
    if source.exists():
        with np.load(source) as data:
            for index, (q, original, estimated) in enumerate(zip(data['indices'], data['original'], data['estimated'])):
                fig, ax = plt.subplots(figsize=(6.4, 4.2))
                ax.plot(original, label='Original', linewidth=1.)
                ax.plot(estimated, label='Estimated', linewidth=1., linestyle='--')
                ax.set_xlabel(r'Device index $i$')
                ax.set_ylabel(r'Effective-channel row energy $\|z_i\|_2^2$')
                cell = ''
                if 'conditions' in data:
                    L, snr = data['conditions'][index]
                    cell = f'L={L:g}, SNR={snr:g} dB\n'
                ax.set_title(label + f' · Tuning realization {q}\n' + cell + context)
                ax.set_ylim(bottom=0); ax.grid(True, alpha=.2); ax.legend()
                fig.tight_layout()
                paths.extend(_save(fig, tuning_figure_path(cfg, method,
                    f'{method}_channel_energy_{int(q):04d}'), formats))
    cell_path = tuning_path(cfg, method, 'condition_scores.csv')
    if cell_path.exists():
        cells = pd.read_csv(cell_path)
        for metric, cmap in [('samples', 'Blues'), ('f1', 'viridis')]:
            grid = cells.pivot(index='snr_db', columns='L', values=metric).sort_index().sort_index(axis=1)
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            im = ax.imshow(grid.to_numpy(), origin='lower', aspect='auto', cmap=cmap)
            ax.set_xticks(range(len(grid.columns)), [f'{v:g}' for v in grid.columns])
            ax.set_yticks(range(len(grid.index)), [f'{v:g}' for v in grid.index])
            for y, row in enumerate(grid.to_numpy()):
                for x, value in enumerate(row):
                    ax.text(x, y, f'{value:.3g}', ha='center', va='center')
            ax.set(xlabel='Pilot length L', ylabel='SNR (dB)', title=label + ' pooled tuning ' + metric)
            fig.colorbar(im, ax=ax, label='Assigned realizations' if metric == 'samples' else 'Mean top-S F1')
            paths.extend(_save(fig, tuning_figure_path(cfg, method, method + '_pooled_' + metric), formats))
    return paths
