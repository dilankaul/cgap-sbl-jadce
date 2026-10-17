import json

import matplotlib.pyplot as plt
import pandas as pd

from cgap_sbl_jadce.figures.common import _save
from cgap_sbl_jadce.paths import tuning_path, tuning_figure_path
from cgap_sbl_jadce.variants import parameters, MATH_SYMBOLS


def sample_axes(method, cfg=None):
    names = parameters(method)
    from cgap_sbl_jadce.settings import tuning_settings
    score = 'f1' if cfg and tuning_settings(cfg, method)['selection_mode'] == 'pooled' else 'tau'
    return [('alpha', score)] if len(names) == 1 else [(name, 'alpha') for name in names[1:]]


def plot_spatial_optuna_tuning(cfg, method, formats='png'):
    from cgap_sbl_jadce.figures.estimator_parameter_sweeps import selected_parameter_label
    trials = pd.read_csv(tuning_path(cfg, method, 'trials.csv')).sort_values('trial')
    selected = json.loads(tuning_path(cfg, method, 'selected.json').read_text())
    values, score = selected['parameters'], selected['tuning_scores']
    context = selected_parameter_label(cfg, method).replace(' · ', '\n').replace('κ_g', r'$\kappa_{\mathrm{g}}$')
    paths = []
    fig, ax = plt.subplots(figsize=(7., 5.2))
    ax.plot(trials.trial + 1, trials.f1, 'o', label='Trial F1')
    ax.plot(trials.trial + 1, trials.f1.cummax(), '-', label='Best F1 so far')
    ax.set(xlabel='Trial', ylabel='Mean F1', title=method.upper() + ' TPE search history\n' + context)
    ax.scatter([score['trial'] + 1], [score['f1']], marker='*', s=130, c='red', label='Selected trial', zorder=5)
    ax.legend()
    paths.extend(_save(fig, tuning_figure_path(cfg, method, method + '_optuna_search_history'), formats))
    for index, (x, y) in enumerate(sample_axes(method, cfg)):
        fig, ax = plt.subplots(figsize=(7., 5.2))
        image = ax.scatter(trials[x], trials[y], c=trials.f1, cmap='viridis')
        from cgap_sbl_jadce.settings import tuning_settings
        tune = tuning_settings(cfg, method)
        if tune.get(x + '_log'):
            ax.set_xscale('log')
        if tune.get(y + '_log'):
            ax.set_yscale('log')
        fig.colorbar(image, ax=ax, label='Mean F1')
        ax.set(xlabel='$' + MATH_SYMBOLS.get(x, x) + '$', ylabel='$' + MATH_SYMBOLS.get(y, y) + '$',
               title=method.upper() + ' sampled parameters\n' + context)
        ax.scatter([values[x]], [values.get(y, score.get(y))], marker='*', s=130, c='red', label='Selected trial')
        ax.legend()
        stem = method + '_optuna_parameter_samples' + ('_' + x if index else '')
        paths.extend(_save(fig, tuning_figure_path(cfg, method, stem), formats))
    return paths
