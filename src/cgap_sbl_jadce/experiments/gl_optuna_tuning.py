import pickle

import numpy as np
import optuna
from tqdm.auto import tqdm

from cgap_sbl_jadce.experiments.search import candidate_key, score_candidate
from cgap_sbl_jadce.io import atomic_path, save_csv, save_json
from cgap_sbl_jadce.paths import tuning_path
from cgap_sbl_jadce.settings import range_values, tuning_settings, validate_gl_search


def _snapshot(checkpoint, study, pending):
    state = {'sampler': study.sampler, 'trials': study.get_trials(deepcopy=True), 'pending': pending}
    checkpoint.save('optimizer', {'optuna_version': optuna.__version__},
                    state=np.frombuffer(pickle.dumps(state, protocol=5), dtype=np.uint8))


def _open_study(folder, sampler, trials):
    path = folder / 'study.sqlite3'
    with atomic_path(path) as temporary:
        storage = optuna.storages.RDBStorage('sqlite:///' + str(temporary.resolve()))
        study = optuna.create_study(storage=storage, study_name='gl_tpe', direction='maximize', sampler=sampler,
                                    pruner=optuna.pruners.NopPruner())
        study.add_trials(trials)
        storage.remove_session()
        storage.engine.dispose()
    storage = optuna.storages.RDBStorage('sqlite:///' + str(path.resolve()))
    return optuna.load_study(study_name='gl_tpe', storage=storage, sampler=sampler,
                             pruner=optuna.pruners.NopPruner()), storage


def tune_gl_optuna(cfg, communication, matrices, checkpoint, solve, show_progress=True):
    settings = validate_gl_search(cfg)
    tune = tuning_settings(cfg, 'gl')
    taus = range_values(tune, 'tau')
    folder = tuning_path(cfg, 'gl', '')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    restored = checkpoint.load('optimizer')
    if restored is not None:
        metadata, arrays = restored
        if metadata.get('optuna_version') != optuna.__version__:
            raise ValueError('Optuna version changed; start a fresh tuning run.')
        state = pickle.loads(arrays['state'].tobytes())
        sampler, trials, pending = state['sampler'], state['trials'], state['pending']
    else:
        snr_key = int(round((float(tune['reference_snr_db']) + 1000.) * 1000))
        seed = int(np.random.SeedSequence([settings['optuna_seed'], int(tune['reference_pilot_length']), snr_key]).generate_state(1)[0])
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=settings['startup_trials'], multivariate=True)
        trials, pending = [], None
    study, storage = _open_study(folder, sampler, trials)
    try:
        if not trials:
            study.enqueue_trial({'alpha': settings['alpha_low']})
            _snapshot(checkpoint, study, None)
        distributions = {name: optuna.distributions.FloatDistribution(settings[name + '_low'], settings[name + '_high'])
                         for name in ('alpha',)}
        completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
        progress = {'completed': completed, 'total': settings['trials'],
                    'samples': settings['samples']}
        save_json(folder / 'search_progress.json', progress)
        with tqdm(total=settings['trials'], initial=completed, desc='GL TPE', unit='trial', disable=not show_progress) as bar:
            while completed < settings['trials']:
                if pending is None:
                    trial = study.ask(fixed_distributions=distributions)
                    pending = trial.number
                    _snapshot(checkpoint, study, pending)
                frozen = study.get_trials()[pending]
                cached = checkpoint.load(f'trial_{pending:04d}')
                if cached is None:
                    row, curve = score_candidate(cfg, frozen.params, communication, matrices, checkpoint, solve, taus,
                        settings['samples'], show_progress, f'GL trial {pending + 1}', prefix='candidate_' + candidate_key(frozen.params))
                    checkpoint.save(f'trial_{pending:04d}', {'row': {**row, 'trial': pending}, 'curve': curve})
                else:
                    row, curve = cached[0]['row'], cached[0]['curve']
                study.tell(pending, row['f1'])
                pending = None
                completed += 1
                _snapshot(checkpoint, study, None)
                rows = [checkpoint.load(f'trial_{trial.number:04d}')[0]['row'] for trial in study.trials
                          if trial.state == optuna.trial.TrialState.COMPLETE]
                save_csv(folder / 'trials.csv', rows)
                save_json(folder / 'search_progress.json', {**progress, 'completed': completed})
                bar.update(1)
        rows = [checkpoint.load(f'trial_{trial.number:04d}')[0]['row'] for trial in study.trials
                  if trial.state == optuna.trial.TrialState.COMPLETE]
        save_csv(folder / 'trials.csv', rows)
        curves = [checkpoint.load(f'trial_{row["trial"]:04d}')[0]['curve'] for row in rows]
        with atomic_path(folder / 'sampler.pkl') as temporary:
            temporary.write_bytes(pickle.dumps(study.sampler, protocol=5))
        return rows, curves
    finally:
        storage.remove_session()
        storage.engine.dispose()
