"""Selected-sample diagnostics for the saved Task-03 communication datasets."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
from tqdm.auto import tqdm
from cgap_sbl_jadce.paths import result_path, figure_path, task_dir
from cgap_sbl_jadce.communication_dataset import load_communication_sample
from cgap_sbl_jadce.figures.common import _apply_figure_style, _save, selected_realizations



def complex_hsv(matrix, magnitude_max):
    """Phase maps cyclically to hue; magnitude maps linearly to brightness."""
    magnitude = np.abs(matrix)
    hue = np.mod((np.angle(matrix) + np.pi) / (2 * np.pi), 1.)
    value = magnitude / magnitude_max if magnitude_max > 0 else np.zeros_like(magnitude)
    return hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.clip(value, 0, 1)], axis=-1))


def _complex_heatmap(matrix, title, xlabel, ylabel, stem, formats, style,
                     antenna_ticks=False, qpsk=False, panel_titles=None, antenna_columns=False, magnitude_max=None):
    matrices = [matrix] if panel_titles is None else matrix
    maximum = max(float(np.abs(value).max()) for value in matrices) if magnitude_max is None else magnitude_max
    fig, axes = plt.subplots(1, len(matrices) + 1, figsize=(6.5 * len(matrices) + 2.5, 4.8),
                             gridspec_kw={'width_ratios': [5] * len(matrices) + [.15 if qpsk else 1.3]},
                             layout='constrained')
    key = axes[-1]
    for index, (ax, value) in enumerate(zip(axes[:-1], matrices)):
        ax.imshow(complex_hsv(value, maximum), aspect='auto', interpolation='nearest')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if antenna_ticks:
            ax.set_yticks(np.arange(value.shape[0]))
        if antenna_columns:
            ax.set_xticks(np.arange(value.shape[1]))
        if panel_titles is not None:
            ax.set_title(panel_titles[index])
    if qpsk:
        phases = np.pi * np.array([-.75, -.25, .25, .75])
        colors = complex_hsv(np.exp(1j * phases)[:, None], 1.)
        key.imshow(colors, origin='lower', aspect='auto', interpolation='nearest')
        key.set_xticks([])
        key.set_yticks(range(4), [r'$-3\pi/4$', r'$-\pi/4$', r'$\pi/4$', r'$3\pi/4$'])
        key.yaxis.tick_right()
        key.yaxis.set_label_position('right')
        key.set_ylabel('Phase (rad)')
        key.set_title('')
    else:
        # A two-dimensional key explains the joint phase/magnitude encoding.
        phase = np.linspace(-np.pi, np.pi, 257)
        scale = maximum if maximum > 0 else 1.
        magnitude = np.linspace(0, scale, 128)
        key_matrix = magnitude[:, None] * np.exp(1j * phase[None, :])
        key.imshow(complex_hsv(key_matrix, scale), origin='lower', aspect='auto',
                   extent=(-np.pi, np.pi, 0, scale), interpolation='nearest')
        key.set_xticks([-np.pi, 0, np.pi], [r'$-\pi$', r'$0$', r'$\pi$'])
        key.set_yticks([0, scale / 2, scale])
        key.set_xlabel('Phase (rad)'); key.set_ylabel('Magnitude')
        key.set_title('Hue: phase\nBrightness: magnitude')
    fig.suptitle(title, math_fontfamily='stix' if qpsk else None)
    _apply_figure_style(fig, style)
    return _save(fig, stem, formats)



def plot_communication_preview(cfg: dict, formats: str = 'png', show_progress: bool = True) -> list[Path]:
    out = Path(cfg['run']['output_dir'])
    summary = json.loads(result_path(out, 'communication', 'summary.json').read_text())
    figure_style = {'axes.titlesize': 12, 'axes.labelsize': 12,
                    'xtick.labelsize': 10, 'ytick.labelsize': 10,
                    'legend.fontsize': 10.5, 'colorbar.labelsize': 12, 'colorbar.ticksize': 10}
    paths = []
    for label in ('tuning', 'evaluation', 'randomized_evaluation'):
        split = 'evaluation' if label == 'randomized_evaluation' else label
        scenario = 'randomized' if label == 'randomized_evaluation' else 'correlated'
        indices = selected_realizations(cfg, split, summary[split]['samples'])
        for q in tqdm(indices, desc=f'Communication figures ({label})', unit='realization',
                      dynamic_ncols=True,
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                      disable=not show_progress):
            L, snr = summary[split]['preview_condition']
            data = load_communication_sample(task_dir(out, 3) / split, q)
            samples = [data[(name, L, snr)] for name in summary[split]['scenarios']]
            r = data[(scenario, L, snr)]
            context = fr"{label.replace('_', ' ').capitalize()} realization {q}: $S_q={int(r.a.sum())}$, $L={L}$, SNR={snr:g} dB"
            stem = lambda name: figure_path(out, f'03_{label}_{name}_{q:04d}')
            if scenario == 'correlated':
                common_context = context.replace('Evaluation realization', 'Evaluation pair')
                gram = np.abs(r.Theta.conj().T @ r.Theta)
                np.fill_diagonal(gram, 0.)
                fig, ax = plt.subplots(figsize=(6.4, 5.2))
                image = ax.imshow(gram, interpolation='nearest')
                cbar = fig.colorbar(image, ax=ax)
                cbar.set_label(r'$|\boldsymbol{\theta}_i^H\boldsymbol{\theta}_j|$', math_fontfamily='stix')
                ax.set_xlabel(r'Pilot index $j$'); ax.set_ylabel(r'Pilot index $i$')
                ax.set_title('Pilot cross-correlation (diagonal removed)\n' + common_context)
                _apply_figure_style(fig, figure_style); fig.tight_layout()
                paths.extend(_save(fig, stem('pilot_cross_correlation'), formats))

                paths.extend(_complex_heatmap(
                    r.H, r'Physical channel matrix $\mathbf{H}$ (HSV)' + '\n' + common_context,
                    r'Device index $i$', r'BS antenna index $m$', stem('channel_hsv_heatmap'),
                    formats, figure_style, antenna_ticks=True))
                paths.extend(_complex_heatmap(
                    r.Theta, r'QPSK pilot matrix $\boldsymbol{\Theta}$ (HSV)' + '\n' + common_context,
                    r'Device / pilot index $i$', r'Pilot symbol index $\ell$', stem('pilot_hsv_heatmap'),
                    formats, figure_style, qpsk=True))
                paths.extend(_complex_heatmap(
                    r.W / np.sqrt(r.sigma2), r'Standardized noise $\mathbf{W}/\sqrt{\sigma^2}$ (HSV)' + '\n' + common_context,
                    r'BS antenna index $m$', r'Pilot symbol index $\ell$', stem('standardized_noise_hsv_heatmap'),
                    formats, figure_style, antenna_columns=True))
            paths.extend(_complex_heatmap(
                r.Z, r'Effective channel matrix $\mathbf{Z}$ (HSV)' + '\n' + context,
                r'BS antenna index $m$', r'Device index $i$', stem('effective_channel_matrix_hsv_heatmap'),
                formats, figure_style, antenna_columns=True,
                magnitude_max=max(float(np.abs(sample.Z).max()) for sample in samples)))

            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            energy = np.sum(np.abs(r.Z)**2, axis=1)
            ax.plot(np.arange(len(energy)), energy, linewidth=.8, color='#333333')
            ax.scatter(np.flatnonzero(r.a), energy[r.a], s=22, color='forestgreen', label='Active MTD')
            maximum_energy = max(float(np.sum(np.abs(sample.Z)**2, axis=1).max()) for sample in samples)
            ax.set_ylim(0, maximum_energy * 1.05 if maximum_energy > 0 else 1.)
            ax.set_xlabel(r'Device index $i$'); ax.grid(True, alpha=.2); ax.legend()
            ax.set_ylabel(r'Effective-channel row energy $\|\mathbf{z}_i\|_2^2$')
            ax.set_title('Effective-channel row energy\n' + context)
            _apply_figure_style(fig, figure_style); fig.tight_layout()
            paths.extend(_save(fig, stem('effective_channel_row_energy'), formats))

            paths.extend(_complex_heatmap(
                r.Y, r'Received signal matrix $\mathbf{Y}$ (HSV)' + '\n' + context,
                r'BS antenna index $m$', r'Pilot symbol index $\ell$', stem('received_signal_hsv_heatmap'),
                formats, figure_style, antenna_columns=True,
                magnitude_max=max(float(np.abs(sample.Y).max()) for sample in samples)))
    return paths
