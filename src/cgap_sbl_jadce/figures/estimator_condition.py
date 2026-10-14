"""Recorded scores by realization for one algorithm and communication condition."""
import matplotlib.pyplot as plt
import pandas as pd

from cgap_sbl_jadce.paths import run_path
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save


def plot_condition_scores(cfg, method, condition, formats='png'):
    from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
    folder = run_path(cfg, method, condition=condition)
    data = pd.read_csv(folder / 'per_sample.csv').drop_duplicates('q').sort_values('q')
    figure_style = {'axes.titlesize': 12, 'axes.labelsize': 11,
                    'xtick.labelsize': 10, 'ytick.labelsize': 10}
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.8), sharex=True)
    for ax, metric, label in zip(axes, ('f1', 'nmse', 'runtime_s'),
                                ('$F_1$ score', 'NMSE (true active support)', 'Solver runtime (s)')):
        ax.plot(data.q, data[metric], linewidth=1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=.25)
        if metric == 'nmse' and data.nmse.gt(0).any():
            ax.set_yscale('log')
    L, snr = condition
    axes[0].set_title(f'{ESTIMATORS[method].name} Evaluation\n' + fr'$L={L:g}$, $\mathrm{{SNR}}={snr:g}$ dB')
    axes[-1].set_xlabel('Evaluation realization index')
    _apply_figure_style(fig, figure_style)
    fig.tight_layout()
    return _save(fig, folder / 'figures' / 'realization_scores', formats)
