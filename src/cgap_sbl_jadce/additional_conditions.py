"""Optional L/SNR conditions shared across algorithms, outside setup settings."""
import json
import math
from pathlib import Path

from cgap_sbl_jadce.io import save_json
from cgap_sbl_jadce.paths import result_path


def validate_pair(condition):
    if len(condition) != 2:
        raise ValueError('Enter a pilot length and SNR.')
    L, snr = condition
    if type(L) is not int or L < 1:
        raise ValueError('Pilot length must be a positive integer.')
    if isinstance(snr, bool) or not isinstance(snr, (float, int)) or not math.isfinite(snr):
        raise ValueError('SNR must be finite.')
    return L, float(snr)


def _registry_paths(cfg):
    from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
    return [Path(cfg['run']['output_dir']) / 'additional_conditions.json'] + [
        result_path(cfg['run']['output_dir'], method, 'additional_conditions.json') for method in ESTIMATORS]


def additional_conditions(cfg, method=None):
    # Read earlier method-local registrations too, without rewriting saved scientific results.
    pairs = []
    for path in _registry_paths(cfg):
        if path.exists():
            pairs.extend(validate_pair(pair) for pair in json.loads(path.read_text())['conditions'])
    return list(dict.fromkeys(pairs))


def add_condition(cfg, method, condition):
    from cgap_sbl_jadce.experiments.estimators import ESTIMATORS, evaluation_conditions
    if method not in ESTIMATORS:
        raise ValueError('Choose a registered algorithm.')
    pair = validate_pair(condition)
    if pair in {value[:2] for value in evaluation_conditions(cfg)}:
        return pair
    pairs = additional_conditions(cfg)
    if pair not in pairs:
        save_json(_registry_paths(cfg)[0], {'conditions': [list(value) for value in pairs + [pair]]})
    return pair


def remove_condition(cfg, condition):
    """Remove the shared option; preserve existing numerical inputs and result files."""
    pair = validate_pair(condition)
    for path in _registry_paths(cfg):
        if not path.exists():
            continue
        saved = json.loads(path.read_text())['conditions']
        remaining = [value for value in saved if validate_pair(value) != pair]
        if remaining != saved:
            save_json(path, {'conditions': remaining})


def input_splits(cfg, methods):
    from cgap_sbl_jadce.settings import tuning_settings
    from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
    return ['evaluation', 'tuning'] if any(ESTIMATORS[method].tuning_grid and tuning_settings(cfg, method)['selection_mode'] == 'per_condition'
                                         for method in methods) else ['evaluation']


def input_reasons(cfg, method, condition):
    from cgap_sbl_jadce.communication_dataset import validate_communication_dataset
    reasons = []
    for split in input_splits(cfg, [method]):
        try:
            validate_communication_dataset(cfg, split, [condition])
        except (OSError, ValueError, KeyError) as error:
            reasons.append(str(error))
    return reasons
