"""Named experiment definitions, edited before work starts and isolated by folder."""
from copy import deepcopy
import math
from pathlib import Path
import re

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.io import save_yaml


def validate_setup(cfg):
    for group, names in {'run': ['num_tuning_samples', 'num_evaluation_samples'],
                         'system': ['N', 'M'], 'activity': ['V', 'batch_size', 'calibration_samples'],
                         'algorithms': ['K', 'cosamp_max_iter']}.items():
        for name in names:
            value = cfg[group][name]
            if type(value) is not int or value < 1:
                raise ValueError(f'{group}.{name} must be a positive integer')
    if cfg['activity'].get('exact_sparsity') and not float(cfg['system']['S']).is_integer():
        raise ValueError('An exact active count must be an integer')
    if cfg['system']['S'] > cfg['system']['N']:
        raise ValueError('Target active count S cannot exceed N')
    for group, names in {'system': ['R', 'S'], 'activity': ['D', 'calibration_tolerance'],
                         'algorithms': ['gamma_init', 'delta', 'cosamp_tol']}.items():
        for name in names:
            value = cfg[group][name]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{group}.{name} must be finite and positive')
    for group, name, integer, positive in [('system', 'pilot_lengths', True, True), ('system', 'snr_db', False, False),
            ('activity', 'kappa_scan', False, True), ('activity', 'kappa_refinement_steps', False, True),
            ('tuning', 'alpha_values', False, False), ('tuning', 'beta_values', False, False)]:
        values = cfg[group][name]
        if not values or len(values) != len(set(values)) or any(not math.isfinite(v) for v in values):
            raise ValueError(f'{group}.{name} needs distinct finite values')
        if integer and any(type(v) is not int for v in values):
            raise ValueError(f'{name} needs integer values')
        if positive and any(v <= 0 for v in values):
            raise ValueError(f'{name} needs positive values')
        if group == 'tuning' and any(v < 0 for v in values):
            raise ValueError(f'{name} cannot contain negative values')
    if cfg['activity']['kappa_scan'] != sorted(cfg['activity']['kappa_scan']):
        raise ValueError('Kappa scan must be ascending')
    if cfg['activity']['kappa_refinement_steps'] != sorted(cfg['activity']['kappa_refinement_steps'], reverse=True):
        raise ValueError('Kappa refinement steps must decrease')
    seeds = list(cfg['seeds'].values())
    if any(type(v) is not int or v < 0 for v in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Use distinct non-negative integer seeds')
    for split in ('tuning', 'evaluation'):
        indices = cfg['figures'][f'{split}_indices']
        count = cfg['run'][f'num_{split}_samples']
        if indices != 'all' and (not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < count for i in indices)):
            raise ValueError(f'{split} figure indices must be between 0 and {count - 1}, or all')
    if cfg['tuning'].get('selection_mode', 'reference') not in {'reference', 'per_condition'}:
        raise ValueError('Unknown tuning selection policy')
    if type(cfg['evaluation']['pilot_length']) is not int or cfg['evaluation']['pilot_length'] < 1:
        raise ValueError('Evaluation fixed pilot length must be a positive integer')
    if type(cfg['tuning']['reference_pilot_length']) is not int or cfg['tuning']['reference_pilot_length'] < 1:
        raise ValueError('Reference pilot length must be a positive integer')
    for name in ('gamma_distribution_samples', 'convergence_sample_offset'):
        value = cfg['evaluation'].get(name, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a non-negative integer')
    if cfg['evaluation'].get('convergence_sample_offset', 0) >= cfg['run']['num_evaluation_samples']:
        raise ValueError('Convergence index must be within the evaluation realizations')
    if cfg['evaluation'].get('condition_mode', 'sweeps') not in {'sweeps', 'grid'}:
        raise ValueError('Unknown evaluation condition mode')
    if not math.isfinite(cfg['tuning']['reference_snr_db']):
        raise ValueError('Reference SNR must be finite')


def setup_files(project, default):
    paths = [Path(default).resolve()]
    for path in sorted((Path(project) / 'results').glob('*/config.yaml')):
        try:
            if load_config(path).get('experiment', {}).get('name') and path.resolve() not in paths:
                paths.append(path.resolve())
        except (OSError, ValueError):
            continue
    return paths


def create_setup(project, name, base):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 _-]{0,79}', name.strip()):
        raise ValueError('Use a name starting with a letter or digit, followed by letters, digits, spaces, dashes or underscores')
    slug = re.sub(r'[ _]+', '-', name.strip().lower())
    folder = Path(project).resolve() / 'results' / slug
    cfg = deepcopy(base)
    cfg.pop('task_runtime', None)
    cfg.pop('spatial_dependence', None)
    if cfg['activity'].get('kappa_source') == 'calibration':
        cfg['activity'].pop('kappa', None)
    cfg['experiment'] = {'name': name.strip()}
    cfg['run']['output_dir'] = folder.relative_to(Path(project).resolve()).as_posix()
    if cfg['tuning'].get('selection_mode', 'reference') == 'per_condition':
        from cgap_sbl_jadce.experiments.estimators import evaluation_conditions
        cfg['communication'] = {'tuning_conditions': sorted({(L, snr) for L, snr, _ in evaluation_conditions(cfg)} |
             {(cfg['tuning']['reference_pilot_length'], cfg['tuning']['reference_snr_db']),
              (cfg['evaluation']['pilot_length'], cfg['tuning']['reference_snr_db'])})}
        cfg['communication']['tuning_conditions'] = [list(pair) for pair in cfg['communication']['tuning_conditions']]
    else:
        cfg.pop('communication', None)
    validate_setup(cfg)
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / 'config.yaml'
    save_yaml(path, cfg)
    return path


def parse_values(text, integer=False):
    if not text.strip():
        return []
    convert = int if integer else float
    try:
        return [convert(value.strip()) for value in text.split(',')]
    except ValueError as error:
        raise ValueError('Enter comma-separated numbers') from error
