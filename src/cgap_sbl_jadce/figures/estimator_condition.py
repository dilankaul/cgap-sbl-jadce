from cgap_sbl_jadce.labels import display_label
import matplotlib.pyplot as plt
import pandas as pd

from cgap_sbl_jadce.paths import run_path
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save, sweep_limits


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
        ax.set_xlim(sweep_limits(data.q))
        ax.set_ylabel(label)
        ax.grid(True, alpha=.25)
        if metric == 'nmse' and data.nmse.gt(0).any():
            ax.set_yscale('log')
    L, snr = condition
    axes[0].set_title(f'{display_label(ESTIMATORS[method].name)} Evaluation\n' + fr'$L={L:g}$, $\mathrm{{SNR}}={snr:g}$ dB')
    axes[-1].set_xlabel('Evaluation realization index')
    _apply_figure_style(fig, figure_style)
    fig.tight_layout()
    paths = _save(fig, folder / 'figures' / 'realization_scores', formats)
    import json
    import numpy as np
    source = folder / 'selected_channel_energy.npz'
    if source.exists():
        parameters = json.loads((folder / 'summary.json').read_text()).get('parameters', {})
        context = ', '.join(f'{key}={value:g}' for key, value in parameters.items())
        with np.load(source) as data:
            for q, original, estimated in zip(data['indices'], data['original'], data['estimated']):
                fig, ax = plt.subplots(figsize=(6.4, 4.2))
                ax.plot(original, label='Original', linewidth=1.)
                ax.plot(estimated, label='Estimated', linewidth=1., linestyle='--')
                ax.set_xlabel(r'Device index $i$')
                ax.set_ylabel(r'Effective-channel row energy $\|z_i\|_2^2$')
                ax.set_title(f'{display_label(ESTIMATORS[method].name)} · Evaluation realization {q}\n'
                             f'L={L:g}, SNR={snr:g} dB' + ('\n' + context if context else ''))
                ax.set_ylim(bottom=0); ax.grid(True, alpha=.2); ax.legend(); fig.tight_layout()
                paths.extend(_save(fig, folder / 'figures' / f'channel_energy_{int(q):04d}', formats))
    return paths
