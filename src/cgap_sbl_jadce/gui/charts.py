import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.settings import evaluation_settings
from cgap_sbl_jadce.figures.common import sweep_limits
from cgap_sbl_jadce.dataset import load_activity_samples
from cgap_sbl_jadce.communication_dataset import load_communication_sample
from cgap_sbl_jadce.spatial_dependence_dataset import load_spatial_dependence_matrix
from cgap_sbl_jadce.paths import result_path, task_dir, tuning_path
from cgap_sbl_jadce.figures.activity_realizations import activity_probability_grid, _activity_plot_options
from cgap_sbl_jadce.figures.communication_diagnostics import complex_hsv
from cgap_sbl_jadce.experiments.estimators import ESTIMATORS


def _style(fig, title, x=None, y=None, log=False):
    fig.update_layout(template='plotly_white', title=title, height=560,
                      font={'family': 'Inter, system-ui, Arial, sans-serif', 'size': 13, 'color':'#334155'},
                      paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                      colorway=['#0f766e','#2563eb','#e08d32','#8b5cf6'],
                      hoverlabel={'bgcolor':'#172b3a', 'font':{'color':'#ffffff', 'size':12}},
                      margin={'l': 65, 'r': 35, 't': 85, 'b': 65}, hovermode='closest',
                      dragmode='pan', uirevision=title)
    fig.update_xaxes(gridcolor='#edf1f5', zerolinecolor='#dce4eb', linecolor='#dce4eb')
    fig.update_yaxes(gridcolor='#edf1f5', zerolinecolor='#dce4eb', linecolor='#dce4eb')
    if x:
        fig.update_xaxes(title=x)
    if y:
        fig.update_yaxes(title=y)
    if log:
        fig.update_yaxes(type='log')
    return fig


def _line_style(fig, title, x, y, log=False, xlim=None):
    _style(fig, title, x, y, log)
    if xlim is None:
        xlim = sweep_limits([value for trace in fig.data for value in trace.x])
    fig.update_xaxes(range=xlim, autorange=False)
    revision = hashlib.sha256(fig.to_json().encode()).hexdigest()
    fig.update_layout(uirevision=revision)
    return fig


def _square_matrix(fig, shape):
    rows, columns = shape
    fig.layout.xaxis.update(range=[-.5, columns - .5], autorange=False, constrain='domain')
    fig.layout.yaxis.update(range=[rows - .5, -.5], autorange=False, constrain='domain',
                            scaleanchor='x', scaleratio=1)
    return fig


def _activity(cfg, request):
    split, q = request['split'], request['index']
    samples = load_activity_samples(result_path(cfg['run']['output_dir'], 'activity', f'{split}_activity.npz'))
    if type(q) is not int or not 0 <= q < len(samples):
        raise ValueError('Realization index is outside the saved dataset')
    sample = samples[q]
    opts = _activity_plot_options(cfg)
    R, D = opts['R'], opts['D']
    X, Y, probability = activity_probability_grid(sample.event_locations, R, D, opts['kappa'])
    fig = go.Figure(go.Heatmap(x=X[0], y=Y[:, 0], z=probability.filled(np.nan), zmin=0, zmax=1,
        colorscale=[[0, '#ffffff'], [.25, '#f5b5b5'], [.5, '#e87c7c'], [.75, '#d44747'], [1, '#a81818']],
        colorbar={'title': 'Activation probability<br>P<sub>act</sub>(<b>r</b>)'}, hoverongaps=False,
        hovertemplate='x=%{x:.2f} m<br>y=%{y:.2f} m<br>P=%{z:.4f}<extra></extra>'))
    active = sample.a.astype(bool)
    for mask, name, color, size in [(~active, 'Inactive MTD', 'lightsteelblue', 5),
                                  (active, 'Active MTD', 'forestgreen', 8)]:
        fig.add_trace(go.Scatter(x=sample.device_locations[mask, 0], y=sample.device_locations[mask, 1],
            mode='markers', name=name, marker={'color': color, 'size': size},
            customdata=np.column_stack([np.flatnonzero(mask), sample.activation_probabilities[mask]]),
            hovertemplate='MTD %{customdata[0]:.0f}<br>x=%{x:.2f} m, y=%{y:.2f} m<br>Pᵢ=%{customdata[1]:.4f}<extra>%{fullData.name}</extra>'))
    fig.add_trace(go.Scatter(x=sample.event_locations[:, 0], y=sample.event_locations[:, 1],
                            mode='markers', name='Event', marker={'symbol': 'star', 'size': 15, 'color': 'firebrick'}))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', name='BS', marker={'symbol': 'x', 'size': 12, 'color': 'black'}))
    angle = np.linspace(0, 2 * np.pi, 201)
    fig.add_trace(go.Scatter(x=R * np.cos(angle), y=R * np.sin(angle), mode='lines', name='Cell boundary',
                            line={'color': '#333333', 'dash': 'dash'}, hoverinfo='skip'))
    for i, (x, y) in enumerate(sample.event_locations):
        fig.add_trace(go.Scatter(x=x + D * np.cos(angle), y=y + D * np.sin(angle), mode='lines',
            name='Event influence boundary', showlegend=i == 0, legendgroup='events',
            line={'color': '#888888', 'dash': 'dash', 'width': 1}, hoverinfo='skip'))
    title = None if request['category'] == 'paper_activity' else f'Event-Driven Activation<br>{split.capitalize()} Realization {q}: {int(active.sum())} Active MTDs'
    _style(fig, title, '<i>x</i> (m)', '<i>y</i> (m)')
    fig.update_yaxes(scaleanchor='x', scaleratio=1, range=[-1.08 * R, 1.08 * R])
    fig.update_xaxes(range=[-1.08 * R, 1.08 * R])
    fig.update_layout(height=730, legend={'orientation': 'h', 'y': -.15})
    return [('Activity realization', fig)]


def _complex(matrix, title, xlabel, ylabel, qpsk=False):
    maximum = float(np.abs(matrix).max())
    fig = make_subplots(rows=1, cols=2, column_widths=[.83, .17], horizontal_spacing=.13)
    rgb = np.rint(complex_hsv(matrix, maximum) * 255).astype(np.uint8)
    values = np.stack([matrix.real, matrix.imag, np.abs(matrix), np.angle(matrix)], axis=-1)
    fig.add_trace(go.Image(z=rgb, customdata=values,
        hovertemplate='Column %{x}, row %{y}<br>Real=%{customdata[0]:.5g}<br>Imag=%{customdata[1]:.5g}<br>Magnitude=%{customdata[2]:.5g}<br>Phase=%{customdata[3]:.5g} rad<extra></extra>'), 1, 1)
    if qpsk:
        phases = np.pi * np.array([-.75, -.25, .25, .75])
        key = np.rint(complex_hsv(np.exp(1j * phases)[:, None], 1.) * 255).astype(np.uint8)
        fig.add_trace(go.Image(z=key, hoverinfo='skip'), 1, 2)
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_yaxes(tickvals=list(range(4)), ticktext=['−3π/4', '−π/4', 'π/4', '3π/4'], title='Phase (rad)', row=1, col=2)
    else:
        phase = np.linspace(-np.pi, np.pi, 129)
        magnitude = np.linspace(0, maximum if maximum > 0 else 1, 64)
        key = np.rint(complex_hsv(magnitude[:, None] * np.exp(1j * phase[None, :]), magnitude[-1]) * 255).astype(np.uint8)
        fig.add_trace(go.Image(z=key, hoverinfo='skip'), 1, 2)
        fig.update_xaxes(tickvals=[0, 64, 128], ticktext=['−π', '0', 'π'], title='Phase (rad)', row=1, col=2)
        fig.update_yaxes(tickvals=[0, 32, 63], ticktext=['0', f'{magnitude[-1]/2:.3g}', f'{magnitude[-1]:.3g}'],
                         title='Magnitude', autorange=True, row=1, col=2)
    _style(fig, title)
    fig.update_xaxes(title=xlabel, row=1, col=1)
    fig.update_yaxes(title=ylabel, scaleanchor=False, row=1, col=1)
    fig.update_xaxes(constrain='domain', row=1, col=1)
    fig.update_yaxes(scaleanchor=False, row=1, col=2)
    return fig


def _communication(cfg, request):
    split, q = request['split'], request['index']
    L, snr = request['condition']
    data = load_communication_sample(task_dir(cfg['run']['output_dir'], 'communication') / split, q)[(L, snr)]
    context = f'{split.capitalize()} Realization {q}: L={L}, SNR={snr:g} dB'
    figures = []
    gram = np.abs(data.Theta.conj().T @ data.Theta)
    np.fill_diagonal(gram, 0)
    fig = go.Figure(go.Heatmap(z=gram, colorscale='Viridis', colorbar={'title': '|θᵢᴴθⱼ|'},
                             hovertemplate='i=%{y}, j=%{x}<br>|θᵢᴴθⱼ|=%{z:.4g}<extra></extra>'))
    _style(fig, 'Pilot Cross-Correlation Magnitude<br>' + context, 'Pilot index j', 'Pilot index i')
    figures.append(('Pilot cross-correlation', _square_matrix(fig, gram.shape)))
    for name, matrix, title, x, y, qpsk in [
            ('H', data.H, 'Physical Channel Matrix <b>H</b>', 'Device index i', 'BS antenna index m', False),
            ('Theta', data.Theta, 'QPSK Pilot Matrix <b>Θ</b>', 'Device index i', 'Pilot symbol index ℓ', True),
            ('W/σ', data.W / np.sqrt(data.sigma2), 'Standardized Noise <b>W</b>/σ', 'BS antenna index m', 'Pilot symbol index ℓ', False),
            ('Z', data.Z, 'Effective Channel Matrix <b>Z</b>', 'BS antenna index m', 'Device index i', False),
            ('Y', data.Y, 'Received Signal Matrix <b>Y</b>', 'BS antenna index m', 'Pilot symbol index ℓ', False)]:
        figures.append((name + ' magnitude and phase', _complex(matrix, title + '<br>' + context, x, y, qpsk)))
    energy = np.sum(np.abs(data.Z)**2, axis=1)
    fig = go.Figure(go.Scatter(x=np.arange(len(energy)), y=energy, mode='lines', name='Row energy', line={'color': '#333333'}))
    fig.add_trace(go.Scatter(x=np.flatnonzero(data.a), y=energy[data.a.astype(bool)], mode='markers', name='Active MTD', marker={'color': 'forestgreen'}))
    figures.append(('Effective-channel row energy', _line_style(fig, 'Effective Channel Energy by MTD<br>' + context, 'Device index i', '‖zᵢ‖₂²')))
    return figures


def interactive_figures(request):
    cfg = load_config(request['config'])
    category = request['category']
    out = cfg['run']['output_dir']
    if category.endswith('_tune'):
        from cgap_sbl_jadce.gui.previews import tuning_preview_config
        cfg = tuning_preview_config(cfg, category.removesuffix('_tune'), request)
    if category in {'activity', 'paper_activity'}:
        return _activity(cfg, request)
    if category == 'communication':
        return _communication(cfg, request)
    if category == 'spatial':
        C = load_spatial_dependence_matrix(task_dir(out, 'spatial_dependence') / request['split'], request['index']).toarray()
        fig = go.Figure(go.Heatmap(z=C, zmin=0, zmax=1, colorscale='Viridis',
                                  hovertemplate='i=%{y}, j=%{x}<br>Cᵢⱼ=%{z:.5g}<extra></extra>'))
        _style(fig, f"Spatial Dependence Matrix <b>C</b><br>{request['split'].capitalize()} Realization {request['index']}", 'MTD index j', 'MTD index i')
        return [('Spatial dependence', _square_matrix(fig, C.shape))]
    if category == 'active_counts':
        fig = go.Figure()
        for split in ('tuning', 'evaluation'):
            samples = load_activity_samples(result_path(out, 'activity', f'{split}_activity.npz'))
            counts = np.asarray([sample.a.sum() for sample in samples])
            fig.add_trace(go.Histogram(x=counts, histnorm='probability', xbins={'size': 1}, opacity=.55,
                                      name=f'{split.capitalize()} (mean={counts.mean():.2f})'))
        summary = json.loads(result_path(out, 'activity', 'summary.json').read_text())
        fig.add_vline(x=summary.get('target_mean_S', cfg['system']['S']), line_dash='dash', annotation_text='Saved target mean S')
        fig.update_layout(barmode='overlay')
        return [('Active count distribution', _style(fig, 'Distribution of Active MTD Count S', 'Realized active count S', 'Relative frequency'))]
    if category == 'kappa':
        df = pd.read_csv(result_path(out, 'kappa_calibration', 'kappa_sweep.csv')).sort_values('kappa')
        summary = json.loads(result_path(out, 'kappa_calibration', 'summary.json').read_text())
        fig = go.Figure()
        for column, name, symbol, size in [('mean_expected_active', 'Mean ΣᵢPᵢ', 'circle', 9), ('mean_realized_active', 'Mean realized S', 'square', 5)]:
            fig.add_trace(go.Scatter(x=df.kappa, y=df[column], mode='lines+markers', name=name, marker={'symbol': symbol, 'size': size}))
        fig.add_hline(y=summary['target_mean_active'], line_dash='dash', annotation_text='Target mean S')
        fig.add_vline(x=summary['closest_grid_kappa'], line_dash='dot', annotation_text='Selected κ')
        return [('Kappa calibration', _line_style(fig, f"Mean Active Count vs κ<br>D={summary['D']:g} m", 'Decay parameter κ', 'Average active count S'))]
    if category.endswith('_tune'):
        method = category.removesuffix('_tune')
        figures = []
        label = ESTIMATORS[method].name
        if method == 'anc':
            df = pd.read_csv(tuning_path(cfg, method, 'alpha_beta.csv'))
            for metric, title, cmap in [('f1', 'Mean Detection F₁', 'Viridis'), ('nmse', 'Mean Channel Estimation NMSE', 'Magma'), ('tau', 'Selected Threshold τ', 'OrRd')]:
                pivot = df.pivot(index='alpha', columns='beta', values=metric).sort_index().sort_index(axis=1)
                values = pivot.to_numpy()
                color = np.log10(np.where(values > 0, values, np.nan)) if metric == 'nmse' else values
                fig = go.Figure(go.Heatmap(x=pivot.columns, y=pivot.index, z=color, customdata=values,
                    colorscale=cmap, colorbar={'title': 'log₁₀ NMSE' if metric == 'nmse' else metric},
                    xgap=1, ygap=1, hovertemplate='α=%{y}, β=%{x}<br>Value=%{customdata:.5g}<extra></extra>'))
                figures.append((title, _style(fig, label + ' α-β Tuning<br>' + title, 'β', 'α')))
        df = pd.read_csv(tuning_path(cfg, method, 'thresholds.csv')).sort_values('tau')
        best = df.sort_values(['f1', 'recall', 'precision'], ascending=False, kind='stable').iloc[0]
        fig = go.Figure()
        for metric in ('f1', 'precision', 'recall'):
            fig.add_trace(go.Scatter(x=df.tau, y=df[metric], mode='lines', name=metric.capitalize()))
            fig.add_trace(go.Scatter(x=[best.tau], y=[best[metric]], mode='markers', showlegend=False,
                                    marker={'size': 9, 'line': {'color': 'black', 'width': 1}}))
        fig.add_vline(x=best.tau, line_dash='dash', annotation_text=f'Selected τ={best.tau:g}')
        figures.append(('Threshold scores', _line_style(fig, label + ' Detection Scores vs Threshold τ', 'Threshold τ', 'Score')))
        return figures
    if category.endswith('_run') or category == 'comparison':
        method = category.removesuffix('_run') if category != 'comparison' else 'comparison'
        if category.endswith('_run') and request.get('condition'):
            from cgap_sbl_jadce.paths import run_path
            pair = tuple(request['condition'])
            folder = run_path(cfg, method, condition=pair)
            data = pd.read_csv(folder / 'per_sample.csv').drop_duplicates('q').sort_values('q')
            figures = []
            for metric, label in [('f1', 'F₁'), ('nmse', 'NMSE (true active support)'), ('runtime_s', 'Solver runtime (s)')]:
                fig = go.Figure(go.Scatter(x=data.q, y=data[metric], mode='lines', name=ESTIMATORS[method].name))
                title = f'{ESTIMATORS[method].name} {label}<br>L={pair[0]}, SNR={pair[1]:g} dB'
                figures.append((label, _line_style(fig, title, 'Evaluation realization index', label, log=metric == 'nmse')))
            gamma_path = folder / 'gamma_distribution.npz'
            if gamma_path.exists() and json.loads((folder / 'summary.json').read_text()).get('gamma_samples', 0):
                with np.load(gamma_path) as archive:
                    a, gamma = archive['a'].astype(bool), archive['gamma']
                fig = go.Figure()
                for mask, name in [(~a, 'Inactive MTD'), (a, 'Active MTD')]:
                    fig.add_trace(go.Histogram(x=gamma[mask], histnorm='probability density', opacity=.55, nbinsx=50, name=name))
                fig.update_layout(barmode='overlay')
                figures.append(('Gamma distribution', _style(fig, f'{ESTIMATORS[method].name} γ Distribution<br>L={pair[0]}, SNR={pair[1]:g} dB', 'γᵢ', 'Density')))
            return figures
        df = pd.read_csv(result_path(out, method, 'aggregate.csv'))
        figures = []
        nmse_label = 'NMSE (true active support)'
        for algorithm, data in df[df.sweep == 'condition'].groupby('algorithm'):
            for metric, label, cmap in [('f1', 'F₁', 'Viridis'), ('nmse', nmse_label, 'Magma'), ('runtime_s', 'Mean solver runtime (s)', 'Cividis')]:
                pivot = data.pivot(index='L', columns='snr_db', values=metric).sort_index().sort_index(axis=1)
                values = pivot.to_numpy()
                color = np.log10(np.where(values > 0, values, np.nan)) if metric == 'nmse' else values
                fig = go.Figure(go.Heatmap(x=pivot.columns, y=pivot.index, z=color, customdata=values,
                    colorscale=cmap, colorbar={'title': 'log₁₀ NMSE' if metric == 'nmse' else label},
                    hovertemplate='L=%{y}, SNR=%{x} dB<br>Value=%{customdata:.5g}<extra></extra>'))
                figures.append((algorithm + ' ' + label, _style(fig, algorithm + ' ' + label + ' by Condition', 'SNR (dB)', 'Pilot length L')))
        settings = evaluation_settings(cfg)
        limits = {'snr_db': sweep_limits(settings['snr_values']),
                  'L': sweep_limits(settings['pilot_lengths'])}
        for sweep, x, xlabel in [('snr_sweep', 'snr_db', 'SNR (dB)'), ('pilot_sweep', 'L', 'Pilot length L')]:
            data = df[df.sweep == sweep]
            if data.empty:
                continue
            fixed = f"L={data.L.iloc[0]:g}" if x == 'snr_db' else f"SNR={data.snr_db.iloc[0]:g} dB"
            for metric, label in [('f1', 'F₁'), ('nmse', nmse_label), ('runtime_s', 'Mean solver runtime (s)')]:
                fig = go.Figure()
                for algorithm, values in data.groupby('algorithm'):
                    values = values.sort_values(x)
                    fig.add_trace(go.Scatter(x=values[x], y=values[metric], mode='lines+markers', name=algorithm))
                title = label + ' vs ' + xlabel + '<br>Fixed ' + fixed
                figures.append((title, _line_style(fig, title, xlabel, label, log=metric == 'nmse', xlim=limits[x])))
        gamma_path = result_path(out, method, 'gamma_distribution.npz') if method != 'comparison' else None
        if gamma_path and gamma_path.exists() and json.loads(result_path(out, method, 'run_summary.json').read_text()).get('gamma_samples', 0):
            with np.load(gamma_path, allow_pickle=False) as archive:
                a, gamma = archive['a'].astype(bool), archive['gamma']
            fig = go.Figure()
            for mask, name in [(~a, 'Inactive MTD'), (a, 'Active MTD')]:
                fig.add_trace(go.Histogram(x=gamma[mask], histnorm='probability density', opacity=.55, nbinsx=50, name=name))
            fig.update_layout(barmode='overlay')
            figures.append(('Gamma distribution', _style(fig, ESTIMATORS[method].name + ' γ Distribution', 'γᵢ', 'Density')))
        return figures
    if category == 'convergence':
        fig = go.Figure()
        with np.load(result_path(out, 'convergence', 'convergence.npz'), allow_pickle=False) as archive:
            for method, spec in ESTIMATORS.items():
                if f'{method}_nmse_history' in archive:
                    values = archive[f'{method}_nmse_history']
                    iterations = archive[f'{method}_iteration_history'] if f'{method}_iteration_history' in archive else np.arange(1, len(values) + 1)
                    fig.add_trace(go.Scatter(x=iterations, y=values, mode='lines+markers', name=spec.name))
        return [('Convergence', _line_style(fig, 'Convergence at Reference Condition', 'Iteration', 'NMSE (true active support)', log=True))]
    raise ValueError('Unknown figure type')


def source_version(request):
    import hashlib
    cfg = load_config(request['config'])
    category, out = request['category'], cfg['run']['output_dir']
    if category.endswith('_tune'):
        from cgap_sbl_jadce.gui.previews import tuning_preview_config
        cfg = tuning_preview_config(cfg, category.removesuffix('_tune'), request)
    paths = [Path(request['config'])]
    if category in {'activity', 'paper_activity', 'active_counts'}:
        splits = (request['split'],) if category != 'active_counts' else ('tuning', 'evaluation')
        paths += [result_path(out, 'activity', f'{split}_activity.npz') for split in splits]
        paths += [result_path(out, 'activity', 'summary.json')]
    elif category in {'communication', 'spatial'}:
        task_id = 'communication' if category == 'communication' else 'spatial_dependence'
        folder = task_dir(out, task_id) / request['split']
        paths += ([folder / f"{request['index']:04d}" / name for name in ('channels.npz', 'pilots.npz', 'observations.npz')]
                  if task_id == 'communication' else [folder / f"{request['index']:04d}.npz"])
    elif category.endswith('_run') and request.get('condition'):
        from cgap_sbl_jadce.paths import run_path
        paths += [run_path(cfg, category.removesuffix('_run'), name, request['condition'])
                  for name in ('per_sample.csv', 'summary.json', 'gamma_distribution.npz')]
    elif category.endswith('_tune'):
        method = category.removesuffix('_tune')
        paths += [tuning_path(cfg, method, name, request.get('condition')) for name in ('selected.json', 'thresholds.csv', 'alpha_beta.csv')]
    else:
        group = {'kappa': 'kappa_calibration', 'convergence': 'convergence'}.get(category, category.removesuffix('_run'))
        paths += [result_path(out, group, name) for name in ('aggregate.csv', 'summary.json', 'run_summary.json', 'gamma_distribution.npz', 'convergence.npz', 'kappa_sweep.csv')]
    if request['category'] == 'communication' and request.get('condition') is not None:
        from cgap_sbl_jadce.paths import condition_name
        paths.append(task_dir(cfg['run']['output_dir'], 'communication') / request['split'] / 'additional' /
                     condition_name(request['condition']) / f"{request['index']:04d}" / 'inputs.npz')
    identity = [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths if path.exists()]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()
