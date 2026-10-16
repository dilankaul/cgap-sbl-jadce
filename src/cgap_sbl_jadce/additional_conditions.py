import json
import math
from pathlib import Path

from cgap_sbl_jadce.io import save_json


def validate_pair(condition):
    if len(condition) != 2:
        raise ValueError('Enter a pilot length and SNR.')
    L, snr = condition
    if type(L) is not int or L < 1:
        raise ValueError('Pilot length must be a positive integer.')
    if isinstance(snr, bool) or not isinstance(snr, (float, int)) or not math.isfinite(snr):
        raise ValueError('SNR must be finite.')
    return L, float(snr)


def _registry_path(cfg):
    return Path(cfg['run']['output_dir']) / 'additional_conditions.json'


def additional_conditions(cfg, method=None):
    path = _registry_path(cfg)
    return [validate_pair(pair) for pair in json.loads(path.read_text())['conditions']] if path.exists() else []


def add_condition(cfg, method, condition):
    from cgap_sbl_jadce.experiments.estimators import ESTIMATORS, evaluation_conditions
    if method not in ESTIMATORS and method != 'communication':
        raise ValueError('Choose a registered algorithm.')
    pair = validate_pair(condition)
    if pair in {value[:2] for value in evaluation_conditions(cfg)}:
        return pair
    pairs = additional_conditions(cfg)
    if pair not in pairs:
        save_json(_registry_path(cfg), {'conditions': [list(value) for value in pairs + [pair]]})
    return pair


def remove_condition(cfg, condition):
    pair = validate_pair(condition)
    remaining = [value for value in additional_conditions(cfg) if value != pair]
    save_json(_registry_path(cfg), {'conditions': [list(value) for value in remaining]})


def input_splits(cfg, methods):
    return ['evaluation', 'tuning']


def input_reasons(cfg, method, condition):
    from cgap_sbl_jadce.communication_dataset import validate_communication_dataset
    reasons = []
    for split in input_splits(cfg, [method]):
        try:
            validate_communication_dataset(cfg, split, [condition])
        except (OSError, ValueError, KeyError) as error:
            reasons.append(str(error))
    return reasons
