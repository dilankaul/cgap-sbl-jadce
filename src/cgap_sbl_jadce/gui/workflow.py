"""Persistent task-card state derived from numerical results and worker records."""
from dataclasses import dataclass
import json

from cgap_sbl_jadce.experiments.readiness import blocking_reasons
from cgap_sbl_jadce.config import resolve_activity_kappa
from cgap_sbl_jadce.paths import result_path
from cgap_sbl_jadce.experiments.estimators import ESTIMATORS, evaluation_conditions, load_selection, completed_methods

ICONS = {'completed': '✓', 'partial': '◐', 'running': '▶', 'queued': '…', 'paused': 'Ⅱ',
         'failed': '!', 'interrupted': '!', 'stopped': '■', 'ready': '○', 'blocked': '×', 'historical': '◇'}


@dataclass
class CardState:
    status: str
    detail: str = ''
    fraction: float = 0.

    @property
    def label(self):
        return f'{ICONS[self.status]} {self.status.capitalize()}'


def ready_state(cfg, task, action):
    """Ready means the default action can run with its current required inputs."""
    reasons = blocking_reasons(cfg, task, action)
    if reasons:
        return CardState('blocked', reasons[0])
    return CardState('ready', 'Required inputs available')


def numerical_state(cfg, task, action='run'):
    out = cfg['run']['output_dir']
    try:
        if task in ESTIMATORS:
            if action == 'tune':
                pairs = cfg.get('communication', {}).get('tuning_conditions', []) if cfg['tuning'].get('selection_mode') == 'per_condition' else [None]
                done = 0
                for pair in pairs:
                    try:
                        load_selection(cfg, task, pair)
                        done += 1
                    except (OSError, ValueError):
                        pass
                if done:
                    return CardState('completed' if done == len(pairs) else 'partial', f'{done}/{len(pairs)} conditions tuned', done / len(pairs))
                return ready_state(cfg, task, action)
            path = result_path(out, task, 'run_summary.json')
            if path.exists():
                data = json.loads(path.read_text())
                if data.get('status') == 'historical' or data.get('nmse_scope') not in {None, 'true_active_support'}:
                    return CardState('historical', 'Historical results retained')
                if data.get('status') in {'complete', 'partial'}:
                    pairs = [value[:2] for value in data.get('completed_conditions', [])] or None
                    if task not in completed_methods(cfg, pairs):
                        return CardState('blocked', 'Results differ from current inputs or selections')
                    done = len(data.get('completed_conditions', []))
                    total = len(evaluation_conditions(cfg))
                    if data['status'] == 'complete' and not done:
                        done = total
                    return CardState('completed' if done == total else 'partial', f'{done}/{total} condition views complete', done / total)
        path = result_path(out, task, 'summary.json')
        if path.exists():
            data = json.loads(path.read_text())
            if task == 'kappa_calibration':
                resolve_activity_kappa(cfg)
                return CardState('completed', f"Selected κ={data['closest_grid_kappa']:g}", 1.)
            if task == 'activity':
                matches = data.get('N') == cfg['system']['N'] and data.get('R') == cfg['system']['R'] and data.get('D') == cfg['activity']['D']
                if matches and all(data.get(split, {}).get('samples') == cfg['run'][f'num_{split}_samples'] and
                                   result_path(out, 'activity', f'{split}_activity.npz').exists() for split in ('tuning', 'evaluation')):
                    return CardState('completed', 'Tuning and evaluation activity saved', 1.)
            elif data.get('status') == 'complete':
                if task in {'communication', 'spatial_dependence'}:
                    if all(data.get(split, {}).get('samples') == cfg['run'][f'num_{split}_samples'] for split in ('tuning', 'evaluation')):
                        return CardState('completed', 'Tuning and evaluation data saved', 1.)
                else:
                    return CardState('completed', ', '.join(data.get('algorithms', data.get('methods', []))), 1.)
    except (OSError, ValueError, KeyError):
        return CardState('blocked', 'Inputs or results differ from this setup')
    return ready_state(cfg, task, action)


def card_state(cfg, config, task, jobs, action='run'):
    saved = numerical_state(cfg, task, action)
    job = jobs.task_job(config, task, action)
    if job and job['status'] in {'queued', 'running', 'paused', 'failed', 'interrupted', 'stopped'}:
        _, progress = jobs.log(job['id'])
        fraction = progress['done'] / max(1, progress['total']) if progress else saved.fraction
        detail = f"{progress['label']} · {progress['done']}/{progress['total']}" if progress else saved.detail
        return CardState(job['status'], detail, min(1., fraction))
    return saved


def combined(states):
    for status in ('running', 'queued', 'paused', 'failed', 'interrupted', 'stopped'):
        for state in states:
            if state.status == status:
                return state
    complete = sum(state.status == 'completed' for state in states)
    if complete == len(states):
        return CardState('completed', 'All actions completed', 1.)
    if complete or any(state.status == 'partial' for state in states):
        return CardState('partial', f'{complete}/{len(states)} actions complete', sum(state.fraction for state in states) / len(states))
    for status in ('ready', 'blocked', 'historical'):
        for state in states:
            if state.status == status:
                return state
    return CardState('blocked', 'No available actions')
