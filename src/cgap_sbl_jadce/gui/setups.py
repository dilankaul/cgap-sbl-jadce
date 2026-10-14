"""Editable experiment definitions, isolated by folder."""
from copy import deepcopy
import math
from pathlib import Path
import re

from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.io import save_yaml
from cgap_sbl_jadce.settings import grouped_config, tuning_pairs, request_settings, calibration_settings, evaluation_settings, range_values
from cgap_sbl_jadce.compatibility import differences


def validate_setup(cfg):
    cfg = grouped_config(cfg)
    def positive(value, name, integer=False, zero=False):
        if (integer and type(value) is not int or not isinstance(value, (float, int)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0 or not zero and value == 0):
            raise ValueError(f'{name} must be a {"non-negative" if zero else "positive"} {"integer" if integer else "finite number"}')
    for group, keys in [('run', ('num_tuning_samples', 'num_evaluation_samples')), ('system', ('N', 'M')), ('activity', ('V', 'batch_size'))]:
        for key in keys:
            positive(cfg[group][key], f'{group}.{key}', integer=True)
    for key in ('R', 'S'):
        positive(cfg['system'][key], f'system.{key}')
    positive(cfg['activity']['D'], 'activity.D')
    if cfg['system']['S'] > cfg['system']['N']:
        raise ValueError('Target active count S cannot exceed N')
    if cfg['activity'].get('exact_sparsity') and not float(cfg['system']['S']).is_integer():
        raise ValueError('An exact active count must be an integer')
    cal = calibration_settings(cfg)
    for key in ('calibration_samples', 'batch_size'):
        positive(cal[key], key, integer=True)
    if 'calibration_validation_samples' in cal:
        positive(cal['calibration_validation_samples'], 'Validation samples', integer=True)
    positive(cal['calibration_tolerance'], 'Calibration tolerance')
    def sequence(values, label, integer=False, zero=False):
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f'{label} needs distinct values')
        for value in values:
            positive(value, label, integer, zero)
    for key in ('kappa_scan', 'kappa_refinement_steps'):
        sequence(cal[key], key)
    if cal['kappa_scan'] != sorted(cal['kappa_scan']):
        raise ValueError('Kappa scan must be ascending')
    if cal['kappa_refinement_steps'] != sorted(cal['kappa_refinement_steps'], reverse=True):
        raise ValueError('Kappa refinement steps must decrease')
    for method, settings in cfg['algorithms'].items():
        for key, value in settings.get('solver', {}).items():
            positive(value, f'{method}.{key}', integer=key in {'K', 'cosamp_max_iter'})
        tune = settings.get('tuning')
        if not tune:
            continue
        if tune['selection_mode'] not in {'reference', 'per_condition'}:
            raise ValueError(f'Unknown {method} tuning policy')
        positive(tune['reference_pilot_length'], 'Reference pilot length', integer=True)
        if not math.isfinite(tune['reference_snr_db']):
            raise ValueError('Reference SNR must be finite')
        sequence(range_values(tune, 'tau'), f'{method}.tau', zero=True)
        for name in ('alpha', 'beta'):
            if name + '_values' in tune or name + '_start' in tune:
                sequence(range_values(tune, name), f'{method}.{name}', zero=True)
    ref = cfg['communication']['reference']
    positive(ref['reference_pilot_length'], 'Communication reference pilot length', integer=True)
    if not math.isfinite(ref['reference_snr_db']):
        raise ValueError('Communication reference SNR must be finite')
    evaluation = evaluation_settings(cfg)
    positive(evaluation['pilot_length'], 'Evaluation fixed pilot length', integer=True)
    if not math.isfinite(evaluation['snr_db']):
        raise ValueError('Evaluation fixed SNR must be finite')
    if evaluation.get('condition_mode', 'sweeps') == 'sweeps':
        sequence(evaluation['pilot_lengths'], 'Pilot lengths', integer=True)
        if not evaluation['snr_values'] or len(evaluation['snr_values']) != len(set(evaluation['snr_values'])) or any(not math.isfinite(v) for v in evaluation['snr_values']):
            raise ValueError('SNR values need distinct finite values')
    elif evaluation.get('condition_mode', 'sweeps') == 'selected':
        pairs = evaluation.get('conditions', [])
        if not pairs or len({tuple(pair) for pair in pairs}) != len(pairs):
            raise ValueError('Choose distinct L/SNR conditions')
        for L, snr in pairs:
            positive(L, 'Pilot length', integer=True)
            if not math.isfinite(snr):
                raise ValueError('SNR must be finite')
    else:
        raise ValueError('Choose sweeps or selected evaluation conditions')
    seeds = list(cfg['seeds'].values())
    if len(seeds) != 6 or any(type(v) is not int or v < 0 for v in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Use six distinct non-negative integer seeds')
    for split in ('tuning', 'evaluation'):
        indices = cfg['figures'][f'{split}_indices']
        count = cfg['run'][f'num_{split}_samples']
        if indices != 'all' and (not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < count for i in indices)):
            raise ValueError(f'{split} figure indices must be between 0 and {count - 1}, or all')
    for name in ('gamma_distribution_samples', 'convergence_sample_offset'):
        positive(evaluation.get(name, 0), name, integer=True, zero=True)
    if evaluation.get('convergence_sample_offset', 0) >= cfg['run']['num_evaluation_samples']:
        raise ValueError('Convergence index must be within the evaluation realizations')


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
    cfg = grouped_config(base)
    cfg.pop('task_runtime', None)
    cfg.pop('spatial_dependence', None)
    if cfg['activity'].get('kappa_source') == 'calibration':
        cfg['activity'].pop('kappa', None)
    cfg['experiment'] = {'name': name.strip()}
    cfg['run']['output_dir'] = folder.relative_to(Path(project).resolve()).as_posix()
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


def setup_changes(before, after):
    before, after = grouped_config(before), grouped_config(after)
    return differences(before, after)


def update_setup(path, candidate, jobs=None):
    """Atomically save settings without touching any data, results or recovery files."""
    path = Path(path).resolve()
    before = load_config(path)
    cfg = grouped_config(candidate)
    if cfg.get('experiment') != before.get('experiment') or cfg['run']['output_dir'] != before['run']['output_dir']:
        raise ValueError('Keep the setup identity and results location; create a new setup to change them.')
    validate_setup(cfg)
    if jobs:
        for job in jobs.records():
            request = job['request']
            if job['status'] in {'running', 'queued'} and Path(request['config']).resolve() == path:
                if request_settings(before, request) != request_settings(cfg, request):
                    raise ValueError('Pause this task and wait for confirmation before editing its required settings.')
    if cfg.get('experiment', {}).get('name'):
        cfg['run']['output_dir'] = 'results/' + path.parent.name
    save_yaml(path, cfg)
    return setup_changes(before, candidate)


def change_impact(before, after):
    """Describe dependency-specific effects before saving; no scientific task is run."""
    from cgap_sbl_jadce.tasks import TASKS, task_actions
    from cgap_sbl_jadce.gui.workflow import numerical_state
    impacts = []
    for task in TASKS:
        for action in task_actions(task):
            old, new = numerical_state(before, task, action), numerical_state(after, task, action)
            if (old.status, old.detail) != (new.status, new.detail):
                impacts.append({'Task': task_actions(task)[action].description.capitalize(), 'Current': old.status.capitalize(),
                                'After saving': new.status.capitalize(), 'Reason': new.detail})
    return impacts
