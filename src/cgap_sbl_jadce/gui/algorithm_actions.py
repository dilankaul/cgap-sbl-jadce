from dataclasses import dataclass, field
import hashlib
import json
import os
import platform

import numpy as np
import scipy

from cgap_sbl_jadce.checkpoints import scientific_revision
from cgap_sbl_jadce.condition_results import completed_conditions, condition_issue
from cgap_sbl_jadce.config import load_config
from cgap_sbl_jadce.experiments.estimators import (evaluation_conditions, evaluation_definition,
                                                 evaluation_selections, selection_digest)
from cgap_sbl_jadce.paths import run_path
from cgap_sbl_jadce.recovery import load_recoveries, validate_recovery


@dataclass
class CompletionPlan:
    pairs: list
    completed: list = field(default_factory=list)
    resume: list = field(default_factory=list)
    start: list = field(default_factory=list)
    blocked: dict = field(default_factory=dict)

    def selected(self, conditions):
        if conditions is None:
            return self
        pairs = {tuple(pair) for pair in conditions}
        return CompletionPlan([pair for pair in self.pairs if pair in pairs],
                              [pair for pair in self.completed if pair in pairs],
                              [pair for pair in self.resume if pair in pairs],
                              [pair for pair in self.start if pair in pairs],
                              {pair: reason for pair, reason in self.blocked.items() if pair in pairs})

    @property
    def summary(self):
        text = f'{len(self.completed)} completed · {len(self.resume)} to resume · {len(self.start)} to start'
        if self.blocked:
            text += f' · {len(self.blocked)} need a rerun'
        return text

    def label(self, selected=False):
        scope = 'selected' if selected else 'all'
        if self.pairs and len(self.completed) + len(self.blocked) == len(self.pairs):
            return f'Rerun {scope}'
        if len(self.start) == len(self.pairs):
            return 'Start'
        if len(self.resume) == len(self.pairs):
            return 'Resume'
        return f'Complete {scope}'


def _checkpoint_issue(folder, expected):
    files = list(folder.glob('.checkpoints/*/*.npz'))
    if not files:
        return None
    manifests = list(folder.glob('.checkpoints/*/definition.json'))
    for path in manifests:
        try:
            saved = json.loads(path.read_text())
            digest = hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()
            if (saved['definition'] == expected and saved['code'] == scientific_revision() and
                    saved['numpy'] == np.__version__ and saved['scipy'] == scipy.__version__ and
                    saved['machine'] == platform.machine() and saved['threads'] == {
                        key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                                            'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS')} and
                    digest == path.parent.name and any(path.parent.glob('*.npz'))):
                return None
        except (OSError, ValueError, KeyError):
            continue
    return 'Stored progress uses different settings, inputs, code or numerical environment.'


def completion_plan(config, method, conditions=None):
    cfg = load_config(config)
    requested = evaluation_conditions(cfg, conditions, method=method)
    plan = CompletionPlan(list(dict.fromkeys(value[:2] for value in requested)))
    try:
        done = {pair for pair, _, _ in completed_conditions(cfg, method)}
        selections = {}
        for pair in plan.pairs:
            try:
                selections.update(evaluation_selections(cfg, method, evaluation_conditions(cfg, [pair], method=method)))
            except (OSError, ValueError, KeyError):
                pass
    except (OSError, ValueError, KeyError):
        done, selections = set(), {}
    recoveries = load_recoveries(config, method)
    validation = {}
    for pair in plan.pairs:
        if pair in done:
            plan.completed.append(pair)
            continue
        folder = run_path(cfg, method, condition=pair)
        summary_path = folder / 'summary.json'
        if summary_path.exists() and json.loads(summary_path.read_text()).get('status') == 'complete':
            plan.blocked[pair] = condition_issue(cfg, method, pair)
            continue
        pending = next((saved for saved in recoveries if pair in {value[:2] for value in evaluation_conditions(cfg, saved['request'].get('conditions'), method=method)}), None)
        units = list(folder.glob('.checkpoints/*/*.npz'))
        if pending is not None:
            path = pending['path']
            if path not in validation:
                try:
                    validate_recovery(pending['request'])
                    validation[path] = None
                except (OSError, ValueError, KeyError) as error:
                    validation[path] = str(error)
            if validation[path]:
                plan.blocked[pair] = validation[path]
                continue
        if units:
            expected = {'definition': evaluation_definition(cfg, method, condition=pair), 'condition': list(pair),
                        'selection_digest': selection_digest(selections[pair]) if pair in selections else None,
                        'gamma_count': min(cfg['evaluation'].get('gamma_distribution_samples', 0),
                                           cfg['run']['num_evaluation_samples'])}
            issue = _checkpoint_issue(folder, expected)
            if issue:
                plan.blocked[pair] = issue
                continue
            plan.resume.append(pair)
        elif pending is not None:
            plan.resume.append(pair)
        elif (folder / 'summary.json').exists():
            plan.blocked[pair] = 'Saved results are incomplete or differ from the current inputs or parameters.'
        else:
            plan.start.append(pair)
    return plan


def clear_finished_recoveries(config, method):
    from cgap_sbl_jadce.recovery import finish_recovery
    cfg = load_config(config)
    done = {pair for pair, _, _ in completed_conditions(cfg, method)}
    for saved in load_recoveries(config, method):
        try:
            pairs = {value[:2] for value in evaluation_conditions(cfg, saved['request'].get('conditions'), method=method)}
        except (OSError, ValueError, KeyError):
            continue
        if pairs <= done:
            finish_recovery(saved['request'])
