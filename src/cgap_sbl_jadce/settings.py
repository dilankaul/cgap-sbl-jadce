from copy import deepcopy
from decimal import Decimal, InvalidOperation
import math

CALIBRATION_KEYS = ('calibration_samples', 'calibration_validation_samples', 'kappa_scan',
                    'kappa_refinement_steps', 'calibration_tolerance', 'batch_size')


def range_values(settings, name, values_key=None, integer=False):
    keys = [name + '_' + part for part in ('start', 'stop', 'step')]
    if not any(key in settings for key in keys):
        return list(settings.get(values_key or name + '_values', []))
    if not all(key in settings for key in keys):
        raise ValueError(f'{name} needs start, stop and step')
    values = [settings[key] for key in keys]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError(f'{name} range needs finite numbers')
    if integer and any(type(value) is not int for value in values):
        raise ValueError(f'{name} range needs integer values')
    start, stop, step = [Decimal(str(value)) for value in values]
    if step <= 0 or stop < start:
        raise ValueError(f'{name} step must be positive and stop must be at least start')
    intervals = (stop - start) / step
    if intervals != intervals.to_integral_value():
        raise ValueError(f'{name} stop must lie on the configured step')
    convert = int if integer else float
    return [convert(start + step * i) for i in range(int(intervals) + 1)]


def range_parameters(values, name, integer=False):
    if not values:
        return None
    try:
        start, stop = values[0], values[-1]
        step = float(Decimal(str(values[1])) - Decimal(str(start))) if len(values) > 1 else 1.
        convert = int if integer else float
        fields = {name + '_start': convert(start), name + '_stop': convert(stop), name + '_step': convert(step)}
        return fields if range_values(fields, name, integer=integer) == list(values) else None
    except (ValueError, TypeError, InvalidOperation, OverflowError):
        return None


def _group_range(settings, name, values_key=None, integer=False):
    key = values_key or name + '_values'
    if any(name + '_' + part in settings for part in ('start', 'stop', 'step')):
        range_values(settings, name, key, integer)
        settings.pop(key, None)
    elif key in settings:
        fields = range_parameters(settings[key], name, integer)
        if fields is not None:
            settings.pop(key)
            settings.update(fields)


def calibration_settings(cfg):
    settings = {**{key: cfg['activity'][key] for key in CALIBRATION_KEYS if key in cfg['activity']},
                **cfg.get('kappa_calibration', {})}
    settings['kappa_scan'] = range_values(settings, 'kappa', 'kappa_scan')
    for part in ('start', 'stop', 'step'):
        settings.pop('kappa_' + part, None)
    return settings


def reference_settings(cfg):
    old = cfg.get('tuning', {})
    return {**{key: old[key] for key in ('reference_pilot_length', 'reference_snr_db') if key in old},
            **cfg.get('communication', {}).get('reference', {})}


def solver_settings(cfg, method):
    algorithms = cfg.get('algorithms', {})
    keys = ('gamma_init', 'K', 'delta') if method in {'sbl', 'anc'} else ()
    old = {key: algorithms[key] for key in keys if key in algorithms}
    if method == 'cosamp':
        old = {'cosamp_lambda_reg': 1e-2,
               **{key: algorithms[key] for key in ('cosamp_max_iter', 'cosamp_tol', 'cosamp_lambda_reg') if key in algorithms}}
    return {**old, **algorithms.get(method, {}).get('solver', {})}


def tuning_settings(cfg, method):
    old = cfg.get('tuning', {})
    settings = {**reference_settings(cfg), 'selection_mode': old.get('selection_mode', 'reference'),
                'tau_start': 0., 'tau_stop': .2, 'tau_step': .001}
    if method == 'anc':
        settings.update({key: old[key] for key in ('alpha_values', 'beta_values') if key in old})
    settings.update(cfg.get('algorithms', {}).get(method, {}).get('tuning', {}))
    return settings


def evaluation_settings(cfg):
    if cfg['evaluation'].get('condition_mode', 'sweeps') != 'sweeps' or 'conditions' in cfg['evaluation']:
        raise ValueError('Evaluation settings use sweeps; add optional conditions on the Run page.')
    settings = {'pilot_lengths': cfg['system'].get('pilot_lengths', []),
                'snr_values': cfg['system'].get('snr_db', []),
                'snr_db': reference_settings(cfg)['reference_snr_db'], **cfg['evaluation']}
    settings['pilot_lengths'] = range_values(settings, 'pilot_length', 'pilot_lengths', integer=True)
    settings['snr_values'] = range_values(settings, 'snr', 'snr_values')
    return settings


def tuning_pairs(cfg, method=None, include_optional=False):
    methods = [method] if method else [name for name in ('sbl', 'anc') if name in cfg.get('algorithms', {})] or ['sbl', 'anc']
    pairs = set()
    if method is None:
        reference = reference_settings(cfg)
        pairs.add((int(reference['reference_pilot_length']), float(reference['reference_snr_db'])))
    for name in methods:
        tune = tuning_settings(cfg, name)
        pairs.add((int(tune['reference_pilot_length']), float(tune['reference_snr_db'])))
        if tune['selection_mode'] == 'per_condition':
            from cgap_sbl_jadce.experiments.estimators import evaluation_conditions
            pairs.update(value[:2] for value in evaluation_conditions(cfg))
    if include_optional and method and tuning_settings(cfg, method)['selection_mode'] == 'per_condition':
        from cgap_sbl_jadce.additional_conditions import additional_conditions
        pairs.update(additional_conditions(cfg, method))

    return sorted(pairs)


def communication_pairs(cfg, split):
    from cgap_sbl_jadce.experiments.estimators import evaluation_conditions
    evaluation = evaluation_settings(cfg)
    pairs = {value[:2] for value in evaluation_conditions(cfg)}
    pairs.add((int(evaluation['pilot_length']), float(evaluation['snr_db'])))
    pairs.update(tuning_pairs(cfg))
    return sorted(pairs)


def communication_conditions(cfg, conditions=None, include_optional=False):
    from cgap_sbl_jadce.additional_conditions import additional_conditions, validate_pair
    pairs = communication_pairs(cfg, 'evaluation')
    optional = additional_conditions(cfg)
    if conditions is None:
        return sorted(set(pairs + optional)) if include_optional else pairs
    selected = list(dict.fromkeys(validate_pair(pair) for pair in conditions))
    if not selected or not set(selected) <= set(pairs + optional):
        raise ValueError('Choose setup conditions or add optional conditions in the Condition status table first.')
    return selected


def grouped_config(cfg):
    result = deepcopy(cfg)
    result.pop('task_runtime', None)
    result.pop('spatial_dependence', None)
    result['kappa_calibration'] = calibration_settings(cfg)
    _group_range(result['kappa_calibration'], 'kappa', 'kappa_scan')
    for key in CALIBRATION_KEYS:
        if key != 'batch_size':
            result['activity'].pop(key, None)
    result['communication'] = {'reference': reference_settings(cfg)}
    result['evaluation'] = evaluation_settings(cfg)
    _group_range(result['evaluation'], 'pilot_length', 'pilot_lengths', integer=True)
    _group_range(result['evaluation'], 'snr', 'snr_values')
    result['system'].pop('pilot_lengths', None)
    result['system'].pop('snr_db', None)
    result['algorithms'] = {name: {'solver': solver_settings(cfg, name)} for name in ('omp', 'cosamp', 'sbl', 'anc')}
    for name in ('sbl', 'anc'):
        result['algorithms'][name]['tuning'] = tuning_settings(cfg, name)
        for parameter in ('alpha', 'beta'):
            _group_range(result['algorithms'][name]['tuning'], parameter)

    for name, values in cfg.get('algorithms', {}).items():
        if isinstance(values, dict) and name not in result['algorithms']:
            result['algorithms'][name] = deepcopy(values)
    result.pop('tuning', None)
    if result['activity'].get('kappa_source') == 'calibration':
        result['activity'].pop('kappa', None)
    return result


def method_config(cfg, method):
    result = deepcopy(cfg)
    tune = tuning_settings(cfg, method)
    result['communication'] = {**result.get('communication', {}), 'reference': {
        key: tune[key] for key in ('reference_pilot_length', 'reference_snr_db')}}
    return result


def request_settings(cfg, request):
    cfg = grouped_config(cfg)
    task, action = request.get('task'), request.get('action', 'run')
    if not task:
        result = grouped_config(cfg)
        result.pop('experiment', None); result['run'].pop('output_dir', None)
        return result
    system, activity = cfg['system'], cfg['activity']
    model = {key: system[key] for key in ('N', 'R', 'S')}
    model['activity'] = deepcopy(activity)
    if activity.get('kappa_source') == 'calibration':
        model['activity'].pop('kappa', None)
    model['activity'].pop('batch_size', None)
    if task == 'kappa_calibration':
        return {**model, 'calibration': calibration_settings(cfg), 'seeds': {key: cfg['seeds'][key] for key in ('kappa_calibration', 'kappa_validation')}}
    if task == 'activity':
        return {**model, 'samples': {split: cfg['run'][f'num_{split}_samples'] for split in ('tuning', 'evaluation')},
                'seeds': {split: cfg['seeds'][f'activity_{split}'] for split in ('tuning', 'evaluation')}}
    if task in {'communication', 'spatial_dependence'}:
        values = {**model, 'samples': {split: cfg['run'][f'num_{split}_samples'] for split in ('tuning', 'evaluation')}}
        if task == 'communication' and request.get('conditions') is not None:
            from cgap_sbl_jadce.additional_conditions import input_splits
            splits = input_splits(cfg, request.get('methods', []))
            values['samples'] = {split: cfg['run'][f'num_{split}_samples'] for split in splits}
            values.update(M=system['M'], conditions=request['conditions'], splits=splits,
                          seeds={split: cfg['seeds'][f'communication_{split}'] for split in splits})
            return values
        if task == 'communication':
            values.update(M=system['M'], tuning_pairs=communication_pairs(cfg, 'tuning'),
                          evaluation_pairs=communication_pairs(cfg, 'evaluation'),
                          seeds={split: cfg['seeds'][f'communication_{split}'] for split in ('tuning', 'evaluation')})
        return values
    if task in {'omp', 'cosamp', 'sbl', 'anc'}:
        split = 'tuning' if action == 'tune' else 'evaluation'
        tune = tuning_settings(cfg, task) if task in {'sbl', 'anc'} else None
        from cgap_sbl_jadce.experiments.estimators import evaluation_conditions
        pairs = request.get('conditions') or (tuning_pairs(cfg, task) if action == 'tune' else sorted({v[:2] for v in evaluation_conditions(cfg)}))
        return {'solver': solver_settings(cfg, task), 'tuning': tune, 'N': system['N'], 'M': system['M'],
                'samples': cfg['run'][f'num_{split}_samples'], 'conditions': sorted(tuple(v) for v in pairs),
                'seed': cfg['seeds'][f'communication_{split}']}
    if task in {'comparison', 'convergence'}:
        return {'methods': {method: request_settings(cfg, dict(request, task=method)) for method in request.get('methods', ['sbl', 'anc'])},
                'q': request.get('realization_index', cfg['evaluation'].get('convergence_sample_offset', 0))}
    return grouped_config(cfg)
