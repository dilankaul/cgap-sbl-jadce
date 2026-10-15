from dataclasses import dataclass
import json
from pathlib import Path
from textwrap import shorten

from cgap_sbl_jadce.experiments.readiness import blocking_reasons
from cgap_sbl_jadce.file_cache import validation_snapshot, file_version
from cgap_sbl_jadce.config import resolve_activity_kappa
from cgap_sbl_jadce.paths import result_path, run_path, tuning_path
from cgap_sbl_jadce.settings import tuning_pairs
from cgap_sbl_jadce.compatibility import validate_preparation, require_match
from cgap_sbl_jadce.experiments.estimators import ESTIMATORS, evaluation_conditions, load_selection, completed_methods

ICONS = {'completed': '✓', 'partial': '◐', 'running': '▶', 'queued': '…', 'paused': 'Ⅱ',
         'failed': '!', 'interrupted': '!', 'stopped': '■', 'ready': '○', 'blocked': '×', 'incompatible': '!'}


@dataclass
class CardState:
    status: str
    detail: str = ''
    fraction: float = 0.

    @property
    def summary(self):
        messages = {
            'incompatible': 'Saved results need review.',
            'blocked': 'Complete the required prerequisites.',
            'failed': 'Open task to review the error.',
            'interrupted': 'Open task to review unfinished work.',
            'stopped': 'Open task to review saved progress.',
        }
        return messages.get(self.status, shorten(self.detail, width=96, placeholder='…'))

    @property
    def label(self):
        return f'{ICONS[self.status]} {self.status.capitalize()}'


def refresh_results(st, cfg, records, rerun=True):
    from cgap_sbl_jadce.paths import TASK_FOLDERS, task_dir
    paths = [Path(cfg['run']['output_dir']) / 'additional_conditions.json']
    for task in TASK_FOLDERS:
        folder = task_dir(cfg['run']['output_dir'], task)
        paths.extend(folder / name for name in ('summary.json', 'run_summary.json'))
        paths.append(folder / 'tuning' / 'selected.json')
        paths.extend(folder.glob('tuning/L*_snr*dB/selected.json'))
        paths.extend(folder.glob('run/L*_snr*dB/summary.json'))
        paths.extend(folder.glob('run/.resets/[0-9a-f]*.json'))
    versions = []
    for path in paths:
        try:
            versions.append(file_version(path))
        except FileNotFoundError:
            continue
    identity = (tuple((job['id'], job['status']) for job in records), tuple(sorted(versions)))
    previous = st.session_state.get('_workflow_revision')
    st.session_state['_workflow_revision'] = identity
    if rerun and previous is not None and previous != identity:
        st.rerun()


def ready_state(cfg, task, action):
    reasons = blocking_reasons(cfg, task, action)
    if reasons:
        return CardState('blocked', reasons[0])
    return CardState('ready', 'Required inputs available')


@validation_snapshot()
def numerical_state(cfg, task, action='run'):
    out = cfg['run']['output_dir']
    stored = False
    try:
        if task in ESTIMATORS:
            if action == 'tune':
                pairs = tuning_pairs(cfg, task)
                done, issues = 0, []
                for pair in pairs:
                    path = tuning_path(cfg, task, 'selected.json', pair)
                    if path.exists():
                        stored = True
                    try:
                        load_selection(cfg, task, pair)
                        done += 1
                    except (OSError, ValueError, KeyError) as error:
                        if path.exists():
                            issues.append(str(error))
                if issues:
                    return CardState('incompatible', issues[0], done / max(1, len(pairs)))
                if done:
                    return CardState('completed' if done == len(pairs) else 'partial', f'{done}/{len(pairs)} conditions tuned', done / len(pairs))
                return ready_state(cfg, task, action)
            from cgap_sbl_jadce.condition_results import completed_conditions, condition_issue, pending_reset_pairs
            pairs = {value[:2] for value in evaluation_conditions(cfg)}
            records = completed_conditions(cfg, task)
            done = {pair for pair, _, _ in records} & pairs
            from cgap_sbl_jadce.experiments.estimators import evaluation_matches
            incompatible = []
            resetting = pending_reset_pairs(cfg, task)
            for pair in pairs - done - resetting:
                path = run_path(cfg, task, 'summary.json', pair)
                if not path.exists():
                    continue
                saved = json.loads(path.read_text())
                if saved.get('status') == 'complete':
                    incompatible.append(pair)
                else:
                    try:
                        if not evaluation_matches(saved.get('definition', {}), cfg, task, pair):
                            incompatible.append(pair)
                    except (OSError, ValueError, KeyError):
                        incompatible.append(pair)
            if incompatible:
                names = ', '.join(f'L={L}, SNR={snr:g} dB' for L, snr in sorted(incompatible))
                return CardState('incompatible', names + ': ' + condition_issue(cfg, task, sorted(incompatible)[0]), len(done) / max(1, len(pairs)))
            if resetting:
                return CardState('partial', 'Resetting conditions', len(done) / max(1, len(pairs)))
            if done:
                return CardState('completed' if done == pairs else 'partial', f'{len(done)}/{len(pairs)} conditions completed', len(done) / len(pairs))
            if records:
                return ready_state(cfg, task, action)
            path = result_path(out, task, 'run_summary.json')
            if path.exists():
                stored = True
                data = json.loads(path.read_text())
                if data.get('status') == 'partial' and not data.get('completed_pairs'):
                    return ready_state(cfg, task, action)
                if task in completed_methods(cfg):
                    return CardState('completed', 'Evaluation data saved', 1.)
                return CardState('incompatible', 'Saved algorithm results do not match this setup')
            return ready_state(cfg, task, action)
        path = result_path(out, task, 'summary.json')
        stored = path.exists()
        if stored:
            data = json.loads(path.read_text())
            if task == 'kappa_calibration':
                resolve_activity_kappa(cfg)
                return CardState('completed', f"Selected κ={data['closest_grid_kappa']:g}", 1.)
            if task == 'activity':
                for split in ('tuning', 'evaluation'):
                    validate_preparation(cfg, task, split)
                return CardState('completed', 'Tuning and evaluation activity saved', 1.)
            if task in {'communication', 'spatial_dependence'} and data.get('status') == 'complete':
                from cgap_sbl_jadce.communication_dataset import validate_communication_dataset
                from cgap_sbl_jadce.spatial_dependence_dataset import validate_spatial_dependence_dataset
                if task == 'communication':
                    from cgap_sbl_jadce.communication_dataset import communication_definition
                    done = total = 0
                    for split in ('tuning', 'evaluation'):
                        validate_communication_dataset(cfg, split, [])
                        expected = {tuple(pair) for pair in communication_definition(cfg, split)['conditions']}
                        available = {tuple(pair) for pair in data[split]['definition']['conditions']}
                        done += len(expected & available); total += len(expected)
                    if done != total:
                        return CardState('partial', f'{done}/{total} communication conditions saved; generate missing conditions', done / total)
                else:
                    for split in ('tuning', 'evaluation'):
                        validate_spatial_dependence_dataset(cfg, split)
                return CardState('completed', 'Tuning and evaluation data saved', 1.)
            if task in {'comparison', 'convergence'} and data.get('status') == 'complete':
                from cgap_sbl_jadce.experiments.estimators import evaluation_definition
                for method, definitions in data.get('definitions', {}).items():
                    if task == 'convergence':
                        require_match(definitions, evaluation_definition(cfg, method, condition=(data['L'], data['snr_db'])), 'Convergence')
                    else:
                        pairs = list(dict.fromkeys(tuple(value[:2]) for value in data['conditions']))
                        require_match(definitions, [evaluation_definition(cfg, method, condition=pair) for pair in pairs], 'Comparison')
                eligible = completed_methods(cfg, list(dict.fromkeys(tuple(value[:2]) for value in data['conditions']))) if task == 'comparison' else completed_methods(cfg, [(data['L'], data['snr_db'])])
                if not set(data['methods']) <= set(eligible):
                    return CardState('incompatible', 'Source algorithm results no longer match the setup')
                return CardState('completed', ', '.join(data.get('algorithms', data.get('methods', []))), 1.)
    except (OSError, ValueError, KeyError) as error:
        return CardState('incompatible' if stored else 'blocked', str(error))
    return ready_state(cfg, task, action)


def card_state(cfg, config, task, jobs, action='run'):
    saved = numerical_state(cfg, task, action)
    job = jobs.task_job(config, task, action)
    if saved.status == 'incompatible' and (not job or job['status'] not in {'queued', 'running'}):
        return saved
    if job and job['status'] in {'paused', 'failed', 'interrupted', 'stopped'}:
        from cgap_sbl_jadce.gui.jobs import config_digest
        try:
            if config_digest(config, job['request']) != job['config_digest']:
                return CardState('incompatible', 'Unfinished work uses different settings; restore them to Resume.')
        except (OSError, ValueError, KeyError):
            return CardState('incompatible', 'Unfinished work does not match this setup.')
    if job and job['status'] in {'queued', 'running', 'paused', 'failed', 'interrupted', 'stopped'}:
        _, progress = jobs.log(job['id'])
        fraction = (saved.fraction if task in ESTIMATORS else
                    progress['done'] / max(1, progress['total']) if progress else saved.fraction)
        detail = f"{progress['label']} · {progress['done']}/{progress['total']}" if progress else saved.detail
        return CardState(job['status'], detail, min(1., fraction))
    return saved


def combined(states):
    for status in ('running', 'queued', 'paused', 'failed', 'interrupted', 'stopped'):
        for state in states:
            if state.status == status:
                return CardState(state.status, state.detail, sum(part.fraction for part in states) / len(states))
    for state in states:
        if state.status == 'incompatible':
            return state
    complete = sum(state.status == 'completed' for state in states)
    if complete == len(states):
        return CardState('completed', 'All actions completed', 1.)
    if complete or any(state.status == 'partial' for state in states):
        return CardState('partial', f'{complete}/{len(states)} actions complete', sum(state.fraction for state in states) / len(states))
    for status in ('ready', 'blocked'):
        for state in states:
            if state.status == status:
                return state
    return CardState('blocked', 'No available actions')
