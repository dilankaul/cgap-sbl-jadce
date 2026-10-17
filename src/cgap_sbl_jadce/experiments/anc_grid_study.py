import os

from tqdm.auto import tqdm

from cgap_sbl_jadce.checkpoints import Checkpoint, checkpointed, output_hashes
from cgap_sbl_jadce.communication_dataset import load_tuning_communication
from cgap_sbl_jadce.experiments.search import score_candidate
from cgap_sbl_jadce.io import save_csv, save_json
from cgap_sbl_jadce.paths import task_dir
from cgap_sbl_jadce.provenance import save_task_snapshot
from cgap_sbl_jadce.settings import grid_study_settings, range_values
from cgap_sbl_jadce.spatial_dependence_dataset import load_tuning_spatial_dependence


def study_config(cfg):
    from cgap_sbl_jadce.experiments.estimators import condition_config
    settings = grid_study_settings(cfg)
    current = condition_config(cfg, (settings['pilot_length'], settings['snr_db']), 'anc')
    current['algorithms']['anc']['tuning'].update({key: settings[key] for key in ('tau_start', 'tau_stop', 'tau_step')})
    return current


def grid_definition(cfg):
    from cgap_sbl_jadce.experiments.estimators import tuning_definition
    current = study_config(cfg)
    values = tuning_definition(current, 'anc')
    for key in ('parameters', 'search_method', 'optuna_version', 'search_code', 'sample_indices'):
        values.pop(key, None)
    from cgap_sbl_jadce.compatibility import _source_digest
    settings = grid_study_settings(cfg)
    values.update(method='ANC parameter grid study', revision=1, study=settings, samples=settings['samples'],
                  sample_indices=list(range(settings['samples'])),
                  study_code=_source_digest(['experiments/anc_grid_study.py', 'experiments/search.py', 'experiments/tuning.py']))
    return values


@checkpointed
def anc_grid_study_stage(cfg, show_progress=True):
    from cgap_sbl_jadce.experiments.estimators import ESTIMATORS
    from cgap_sbl_jadce.figures.anc_parameter_tuning import plot_anc_grid_study
    settings = grid_study_settings(cfg)
    if not 0 < settings['samples'] <= cfg['run']['num_tuning_samples']:
        raise ValueError('Grid study samples must be within the saved tuning sample count.')
    current = study_config(cfg)
    definition = grid_definition(cfg)
    folder = task_dir(cfg['run']['output_dir'], 'anc_grid_study')
    communication = load_tuning_communication(current, settings['samples'])
    matrices = load_tuning_spatial_dependence(current, settings['samples'])
    parameters = [{'alpha': alpha, 'beta': beta} for alpha in range_values(settings, 'alpha')
                  for beta in range_values(settings, 'beta')]
    checkpoint = Checkpoint(folder, {'task': 'anc_grid_study', 'definition': definition})
    save_json(folder / 'summary.json', {'status': 'started', 'definition': definition})
    save_task_snapshot(cfg, 'anc_grid_study', status='started')
    rows, scores = [], []
    for index, values in enumerate(tqdm(parameters, desc='ANC paper grid', unit='pair', disable=not show_progress)):
        cached = checkpoint.load(f'pair_{index:04d}')
        if cached is None:
            row, curve = score_candidate(current, values, communication, matrices, checkpoint,
                ESTIMATORS['anc'].solve, range_values(settings, 'tau'), settings['samples'], show_progress,
                f'Grid pair {index + 1}')
            checkpoint.save(f'pair_{index:04d}', {'row': row, 'curve': curve})
        else:
            row, curve = cached[0]['row'], cached[0]['curve']
        rows.append(row)
        scores.extend({**values, **point} for point in curve)
        save_json(folder / 'progress.json', {'completed': index + 1, 'total': len(parameters)})
    save_csv(folder / 'alpha_beta.csv', rows)
    save_csv(folder / 'alpha_beta_tau_scores.csv', scores)
    plots = plot_anc_grid_study(cfg)
    files = [folder / name for name in ('alpha_beta.csv', 'alpha_beta_tau_scores.csv')] + plots
    summary = {'status': 'complete', 'definition': definition, 'pairs': len(parameters),
               'outputs': {os.path.relpath(path, folder): digest for path, digest in zip(files, output_hashes(files).values())}}
    save_json(folder / 'summary.json', summary)
    save_task_snapshot(cfg, 'anc_grid_study', status='complete')
    return summary
