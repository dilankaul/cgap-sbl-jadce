"""Descriptive output folders owned by independent workflow tasks."""
from pathlib import Path

TASK_FOLDERS = {
    'kappa_calibration': 'kappa_calibration', 'activity': 'activity',
    'communication': 'communication', 'spatial_dependence': 'spatial_dependence',
    'omp': 'omp', 'cosamp': 'cosamp', 'sbl': 'sbl', 'anc': 'cgap_sbl_anc',
    'comparison': 'comparison', 'convergence': 'convergence',
}


def task_dir(out: str | Path, task: str) -> Path:
    return Path(out) / TASK_FOLDERS[task]


def result_path(out: str | Path, group: str, filename: str) -> Path:
    folder = task_dir(out, group)
    if group == 'activity' and filename in {'tuning_activity.npz', 'evaluation_activity.npz'}:
        folder /= filename.removesuffix('_activity.npz')
    return folder / filename


def figure_path(out, task, stem, split=None, index=None, section=None):
    folder = task_dir(out, task) / 'figures'
    if split is not None and index is not None:
        folder /= f'{split}_{index:04d}'
    elif section:
        folder /= section
    return folder / stem


def tuning_path(cfg, method, filename, condition=None):
    folder = task_dir(cfg['run']['output_dir'], method) / 'tuning'
    from cgap_sbl_jadce.settings import tuning_settings
    tune = tuning_settings(cfg, method)
    if tune['selection_mode'] == 'per_condition':
        L, snr = condition or (tune['reference_pilot_length'], tune['reference_snr_db'])
        folder /= f'L{int(L):04d}_snr{float(snr):g}dB'
    return folder / filename


def tuning_figure_path(cfg, method, stem):
    folder = task_dir(cfg['run']['output_dir'], method) / 'figures' / 'tuning'
    source = tuning_path(cfg, method, '')
    if source.name != 'tuning':
        folder /= source.name
    return folder / stem


def condition_name(condition):
    L, snr = condition
    return f'L{int(L):04d}_snr{float(snr):g}dB'


def run_path(cfg, method, filename='', condition=None):
    folder = task_dir(cfg['run']['output_dir'], method) / 'run'
    if condition is not None:
        folder /= condition_name(condition)
    return folder / filename
