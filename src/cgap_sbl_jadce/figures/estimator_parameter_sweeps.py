from cgap_sbl_jadce.variants import SPATIAL_METHODS, parameters, SYMBOLS, MATH_SYMBOLS
import json

import numpy as np
import matplotlib.pyplot as plt

from cgap_sbl_jadce.figures.common import _save, figure_formats, sweep_limits
from cgap_sbl_jadce.paths import task_dir
from cgap_sbl_jadce.settings import evaluation_settings, tuning_settings


def parameter_sweeps(cfg, method):
    from cgap_sbl_jadce.experiments.estimators import tuning_conditions
    from cgap_sbl_jadce.tuning_results import completed_tuning_selection
    if tuning_settings(cfg, method)['selection_mode'] != 'per_condition':
        return []
    selected = {}
    for pair in tuning_conditions(cfg, method=method):
        try:
            selected[pair] = completed_tuning_selection(cfg, method, pair)['parameters']
        except (OSError, ValueError, KeyError):
            continue
    if not selected:
        return []
    settings = evaluation_settings(cfg)
    rows = []
    for axis, values, pairs, xlabel, fixed in (
        ('snr', settings['snr_values'], [(settings['pilot_length'], snr) for snr in settings['snr_values']],
         'SNR (dB)', f"Fixed L={settings['pilot_length']}"),
        ('pilot_length', settings['pilot_lengths'], [(L, settings['snr_db']) for L in settings['pilot_lengths']],
         'Pilot length L', f"Fixed SNR={settings['snr_db']:g} dB")):
        for name, symbol in [(name, SYMBOLS[name]) for name in (*parameters(method), 'tau')]:
            rows.append({'parameter': name, 'symbol': symbol, 'axis': axis, 'x': list(values),
                         'y': [selected.get(pair, {}).get(name, np.nan) for pair in pairs],
                         'xlabel': xlabel, 'fixed': fixed, 'label': f'{symbol} vs {xlabel}'})
    return rows


def sweep_figure_dir(cfg, method):
    return task_dir(cfg['run']['output_dir'], method) / 'figures' / 'tuning' / 'sweeps'


def plot_tuning_parameter_sweeps(cfg, method, formats='png'):
    curves = parameter_sweeps(cfg, method)
    folder = sweep_figure_dir(cfg, method)
    if not curves:
        for name in (*parameters(method), 'tau'):
            for axis in ('snr', 'pilot_length'):
                for suffix in figure_formats(formats):
                    (folder / f'{method}_{name}_vs_{axis}.{suffix}').unlink(missing_ok=True)
        return []
    paths = []
    for curve in curves:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(curve['x'], curve['y'], 'o-')
        for x, y in zip(curve['x'], curve['y']):
            if np.isfinite(y):
                ax.annotate(f'{y:.6g}', (x, y), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
        ax.set(xlabel=curve['xlabel'], ylabel="Selected $" + MATH_SYMBOLS[curve['parameter']] + "$",
               title=curve['label'].replace('κ_g', r'$\kappa_{\mathrm{g}}$') + '\n' + curve['fixed'], xlim=sweep_limits(curve['x']))
        ax.grid(True, alpha=.25)
        paths.extend(_save(fig, folder / f"{method}_{curve['parameter']}_vs_{curve['axis']}", formats))
    return paths


def selected_parameter_label(cfg, method):
    from cgap_sbl_jadce.paths import tuning_path
    path = tuning_path(cfg, method, 'selected.json')
    if method not in SPATIAL_METHODS or not path.exists():
        return ''
    selected = json.loads(path.read_text())
    p = selected['parameters']
    text = 'Selected ' + ', '.join(f'{SYMBOLS[name]}={p[name]:.6g}' for name in (*parameters(method), 'tau'))
    if selected.get('definition', {}).get('objective') == 'mean_top_S_gamma_f1':
        return text + ' · pooled top-S selection; τ calibrated afterward'
    condition = selected.get('definition', {}).get('condition')
    if condition:
        text += f" · L={condition[0]:g}, SNR={condition[1]:g} dB"
    return text
