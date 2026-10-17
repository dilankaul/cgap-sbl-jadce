import json

import matplotlib.pyplot as plt
import pandas as pd

from cgap_sbl_jadce.figures.common import _save
from cgap_sbl_jadce.paths import tuning_path, tuning_figure_path


def plot_anc_optuna_tuning(cfg, formats='png'):
    trials = pd.read_csv(tuning_path(cfg, 'anc', 'trials.csv')).sort_values('trial')
    from cgap_sbl_jadce.figures.estimator_parameter_sweeps import selected_parameter_label
    selected = json.loads(tuning_path(cfg, 'anc', 'selected.json').read_text())
    parameters = selected['parameters']
    context = selected_parameter_label(cfg, 'anc').replace(' · ', '\n')
    paths = []
    fig, ax = plt.subplots(figsize=(7., 5.2))
    ax.plot(trials.trial + 1, trials.f1, 'o', label='Trial F1')
    ax.plot(trials.trial + 1, trials.f1.cummax(), '-', label='Best F1 so far')
    ax.set(xlabel='Trial', ylabel='Mean F1', title='ANC TPE search history\n' + context)
    score = selected['tuning_scores']
    ax.scatter([score['trial'] + 1], [score['f1']], marker='*', s=130, c='red', label='Selected trial', zorder=5)
    ax.legend()
    paths.extend(_save(fig, tuning_figure_path(cfg, 'anc', 'anc_optuna_search_history'), formats))
    fig, ax = plt.subplots(figsize=(7., 5.2))
    image = ax.scatter(trials.beta, trials.alpha, c=trials.f1, cmap='viridis')
    fig.colorbar(image, ax=ax, label='Mean F1')
    ax.set(xlabel=r'$\beta$', ylabel=r'$\alpha$', title='ANC sampled parameter pairs\n' + context)
    ax.scatter([parameters['beta']], [parameters['alpha']], marker='*', s=130, c='red', label='Selected pair')
    ax.legend()
    paths.extend(_save(fig, tuning_figure_path(cfg, 'anc', 'anc_optuna_parameter_samples'), formats))
    return paths
