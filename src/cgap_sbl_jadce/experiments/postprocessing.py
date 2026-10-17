"""Detection and channel-error analysis from final numerical estimates."""
import numpy as np
import pandas as pd

from cgap_sbl_jadce.metrics import nmse, precision_recall_f1, support_from_scores, support_from_rows
from cgap_sbl_jadce.settings import analysis_settings
from cgap_sbl_jadce.paths import run_path


def detection_support(Z_hat, gamma, settings, selection=None, native_budget=None):
    selection = selection or {}
    mode = settings['detection_mode']
    row_norm = np.linalg.norm(Z_hat, axis=1) / np.sqrt(Z_hat.shape[1])
    calibration = selection.get('calibration', {})
    if mode == 'gamma':
        if gamma is None:
            if native_budget is None:
                raise ValueError('Gamma detection requires an SBL-family estimate.')
            return support_from_rows(Z_hat, native_budget)
        threshold = settings['gamma_threshold']
        if threshold is None:
            threshold = calibration.get('gamma_threshold', selection.get('parameters', {}).get('tau'))
        scores = gamma
    elif mode == 'z':
        threshold = settings['z_threshold']
        if threshold is None:
            threshold = calibration.get('z_threshold')
        scores = row_norm
    else:
        scores = gamma if settings['top_k_score'] == 'gamma' and gamma is not None else row_norm
        return support_from_scores(scores, settings['top_k'])
    if threshold is None:
        raise ValueError('Choose an explicit threshold or prepare a compatible tuning calibration.')
    return np.asarray(scores) >= threshold


def rescore_condition(cfg, method, pair, frame, selection=None, show_progress=True):
    from cgap_sbl_jadce.communication_dataset import validate_communication_dataset, load_communication_sample
    settings = analysis_settings(cfg, method)
    source = run_path(cfg, method, 'final_estimates.npz', pair)
    with np.load(source, allow_pickle=False) as data:
        if not np.array_equal(data['q'], np.arange(cfg['run']['num_evaluation_samples'])):
            raise ValueError('Final estimate realization indices are incomplete.')
        estimates = data['Z_hat']
        gammas = data['gamma'] if 'gamma' in data else None
        masks = data['a']
    table = frame.drop_duplicates('q').set_index('q')
    communication = validate_communication_dataset(cfg, 'evaluation', [pair])
    rows = []
    from tqdm.auto import tqdm
    from cgap_sbl_jadce.execution_control import pause_point
    for q in tqdm(range(len(estimates)), desc=f'{method.upper()} analysis L={pair[0]} SNR={pair[1]:g} dB',
                  unit='realization', disable=not show_progress):
        pause_point()
        estimate = estimates[q]
        old = table.loc[q].to_dict()
        support = detection_support(estimate, gammas[q] if gammas is not None else None, settings,
                                    selection, int(old['B_q']) if 'B_q' in old else None)
        precision, recall, f1 = precision_recall_f1(masks[q], support)
        truth = load_communication_sample(communication, q, pair)[pair]
        if not np.array_equal(truth.a, masks[q]):
            raise ValueError('Final estimates do not match the saved activity inputs.')
        unmasked = nmse(truth.Z, estimate, truth.a)
        masked = nmse(truth.Z, estimate * support[:, None], truth.a)
        rows.append({**old, 'q': q, 'precision': precision, 'recall': recall, 'f1': f1,
                     'nmse': masked if settings['nmse_mode'] == 'support_masked' else unmasked,
                     'nmse_unmasked': unmasked, 'nmse_support_masked': masked,
                     'nmse_mode': settings['nmse_mode'], 'detection_mode': settings['detection_mode'] if gammas is not None or settings['detection_mode'] != 'gamma' else 'native_oracle'})
    rescored = pd.DataFrame(rows).drop(columns='sweep')
    return pd.concat([rescored.assign(sweep=view) for view in frame.sweep.unique()], ignore_index=True)


def analysis_revision():
    from cgap_sbl_jadce.compatibility import _source_digest
    return _source_digest(['experiments/postprocessing.py', 'metrics.py'])
